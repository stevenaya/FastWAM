"""CPU-only RTC sampler/codec tests, without weights, GPU, or a policy server."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_idm import FastWAMIDM
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.models.wan22.fastwam_optional_idm import FastWAMOptionalIDM
from fastwam.models.wan22.rtc import ActionPriorInpainting
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from fastwam.openarm import RelativeArmJoints, decode_actions, encode_actions, encode_state
from scripts.serve_openarm import OpenArmPolicy


HORIZON = 5
ARM_INDICES = list(range(7)) + list(range(8, 15))


def velocity(x, timestep):
    return x * 0.25 + timestep[:, None, None] * 0.0001


class TinyFastWAM(FastWAM):
    """Exercise the real infer_action with cheap deterministic tensor experts."""

    def __init__(self, dtype=torch.float32):
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.torch_dtype = dtype
        self.proprio_dim = 16
        self.proprio_encoder = None
        self.action_expert = SimpleNamespace(action_dim=16)
        self.video_expert = SimpleNamespace(
            video_attention_mask_mode="first_frame_causal", prepare=self.prepare_video,
            build_video_to_video_mask=lambda **kw: torch.ones(2, 2, dtype=torch.bool),
        )
        self.mot = SimpleNamespace(prefill_video_cache_tensor=lambda **kw: ([torch.zeros(1)], [torch.zeros(1)]))
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
        self.seen = []
        self.predict_velocity = velocity

    def _encode_input_image_latents_tensor(self, **kwargs):
        return torch.zeros(1, 1, 1, 1, 1, dtype=self.torch_dtype)

    def prepare_video(self, **kwargs):
        return (torch.zeros(1, 2, 4), None, None, kwargs["context"],
                kwargs["context_mask"], None, None, None, None, 2)

    def _denoise_action_with_video_cache(self, **kwargs):
        self.seen.append({key: value.clone() if torch.is_tensor(value) else value
                          for key, value in kwargs.items()})
        return self.predict_velocity(kwargs["latents_action"], kwargs["timestep_action"])


def sample(model, **kwargs):
    options = dict(prompt=None, input_image=torch.zeros(3, 16, 16), action_horizon=HORIZON,
                   context=torch.zeros(1, 2, 4), context_mask=torch.ones(1, 2, dtype=torch.bool),
                   seed=42, num_inference_steps=4)
    options.update(kwargs)
    return model.infer_action(**options)["action"]


def processor():
    norm = SingleFieldLinearNormalizer(
        {"mean": torch.linspace(-0.2, 0.2, 16), "std": torch.linspace(0.1, 0.5, 16)}, mode="z-score"
    )
    return SimpleNamespace(
        normalizer=SimpleNamespace(normalizers={"action": {"default": norm}, "state": {"default": norm}}),
        action_state_transforms=[RelativeArmJoints(ARM_INDICES)],
    )


def policy(model):
    result = OpenArmPolicy.__new__(OpenArmPolicy)
    result.model = model
    result.processor = processor()
    result.cfg = SimpleNamespace(data=SimpleNamespace(train=SimpleNamespace(num_frames=HORIZON + 1)))
    result.steps = 4
    result.seed = 42
    result.compile_action_infer = False
    result.checkpoint_step = 1
    result.contexts = {DEFAULT_PROMPT.format(task="pillow"): (torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool))}
    return result


class RTCSamplerTests(unittest.TestCase):
    def test_projection_weights_form_a_continuous_ramp(self):
        prior = torch.full((HORIZON, 16), -2.)
        noise = torch.full((1, HORIZON, 16), 1.)
        weights = torch.linspace(0, 1, HORIZON)
        rtc = ActionPriorInpainting(prior, weights, noise)
        proposal = torch.full_like(noise, 3.)
        result = rtc.constrain(proposal, torch.tensor(0.5))
        expected = -0.5 * (1 - weights) + 3 * weights
        torch.testing.assert_close(result[0, :, 0], expected)
        self.assertTrue((result[0, 1:, 0] > result[0, :-1, 0]).all())
        self.assertTrue(torch.equal(result[0, -1], proposal[0, -1]))

    def test_off_and_zero_constraint_match_legacy_loop_and_rng(self):
        for dtype in (torch.float32, torch.bfloat16):
            for seed in (None, 42):
                with self.subTest(dtype=dtype, seed=seed):
                    torch.manual_seed(123)
                    generator = None if seed is None else torch.Generator().manual_seed(seed)
                    x = torch.randn(1, HORIZON, 16, generator=generator).to(dtype)
                    scheduler = WanContinuousFlowMatchScheduler(shift=5.0)
                    times, deltas = scheduler.build_inference_schedule(4, torch.device("cpu"), dtype)
                    for timestep, delta in zip(times, deltas):
                        x = scheduler.step(velocity(x, timestep.unsqueeze(0)), delta, x)
                    expected_rng = torch.get_rng_state()
                    for rtc in ({}, {"action_prior": torch.full((HORIZON, 16), 19.3),
                                     "action_update_weights": torch.ones(HORIZON)}):
                        torch.manual_seed(123)
                        output = sample(TinyFastWAM(dtype), seed=seed, **rtc)
                        self.assertTrue(torch.equal(output, x[0].float()))
                        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_frozen_prefix_is_anchored_at_every_sigma_and_exact_endpoint(self):
        prior = torch.linspace(-12.73, 9.23, HORIZON * 16).reshape(HORIZON, 16)
        for dtype in (torch.float32, torch.bfloat16):
            model = TinyFastWAM(dtype)
            output = sample(model, action_prior=prior, action_update_weights=torch.tensor([0., 0., .2, .7, 1.]))
            noise = model.seen[0]["latents_action"].float()
            for call in model.seen:
                sigma = call["timestep_action"].float().reshape(()) / 1000
                anchor = torch.lerp(prior.unsqueeze(0), noise, sigma).to(dtype)
                self.assertTrue(torch.equal(call["latents_action"][:, :2], anchor[:, :2]))
                self.assertEqual(call["latents_action"].dtype, dtype)
                self.assertEqual(call["latents_action"].shape, (1, HORIZON, 16))
            self.assertTrue(torch.equal(output[:2], prior[:2]))

    def test_default_weights_freeze_all_batched_prior(self):
        prior = torch.full((1, HORIZON, 16), 0.1234567)
        self.assertTrue(torch.equal(sample(TinyFastWAM(torch.bfloat16), action_prior=prior), prior[0]))

    def test_ramp_matches_per_step_soft_projection_and_free_tail(self):
        prior = torch.linspace(-1, 1, HORIZON * 16).reshape(HORIZON, 16)
        weights = torch.tensor([0., .1, .4, .8, 1.])
        model = TinyFastWAM()
        result = sample(model, action_prior=prior, action_update_weights=weights, sigma_shift=3.7)
        times, deltas = model.infer_action_scheduler.build_inference_schedule(
            4, torch.device("cpu"), torch.float32, shift_override=3.7
        )
        noise = model.seen[0]["latents_action"]
        x = noise.clone()
        for i, (t, dt) in enumerate(zip(times, deltas)):
            torch.testing.assert_close(model.seen[i]["latents_action"], x)
            x = x + velocity(x, t.unsqueeze(0)) * dt
            next_sigma = times[i + 1] / 1000 if i + 1 < len(times) else torch.tensor(0.)
            anchor = prior * (1 - next_sigma) + noise * next_sigma
            x = x * weights[None, :, None] + anchor * (1 - weights[None, :, None])
        torch.testing.assert_close(result, x[0])
        baseline = sample(TinyFastWAM(), sigma_shift=3.7)
        self.assertTrue(torch.equal(result[-1], baseline[-1]))

    def test_nonlinear_schedule_has_negative_dt_and_noise_minus_action_velocity(self):
        model = TinyFastWAM()
        target = torch.full((1, HORIZON, 16), 0.25)
        model.predict_velocity = lambda x, t: model.seen[0]["latents_action"] - target
        output = sample(model, action_prior=torch.ones(HORIZON, 16), action_update_weights=torch.ones(HORIZON))
        times, deltas = model.infer_action_scheduler.build_inference_schedule(4, torch.device("cpu"), torch.float32)
        self.assertTrue((deltas < 0).all())
        self.assertGreater(float(deltas.std()), 0.01)
        torch.testing.assert_close(deltas.sum(), torch.tensor(-1.))
        torch.testing.assert_close(output, target[0], atol=1e-6, rtol=0)
        sigma = times[1] / 1000
        noise = model.seen[0]["latents_action"]
        torch.testing.assert_close(model.infer_action_scheduler.add_noise(target, noise, times[1]),
                                   target * (1 - sigma) + noise * sigma)
        self.assertTrue(torch.equal(model.infer_action_scheduler.training_target(target, noise, times[1]), noise - target))

    def test_compiled_entry_shapes_and_arguments_unchanged(self):
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as compile_fn, \
                patch("torch.compiler.cudagraph_mark_step_begin"):
            base = TinyFastWAM(torch.bfloat16)
            sample(base, compile_action_infer=True)
            rtc = TinyFastWAM(torch.bfloat16)
            result = sample(rtc, compile_action_infer=True, action_prior=torch.zeros(HORIZON, 16))
        self.assertEqual(compile_fn.call_count, 4)
        self.assertEqual(set(base.seen[0]), set(rtc.seen[0]))
        for call in rtc.seen:
            self.assertEqual(call["latents_action"].dtype, torch.bfloat16)
            self.assertEqual(call["latents_action"].shape, (1, HORIZON, 16))
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_useful_input_errors(self):
        for kwargs in (
            {"action_update_weights": torch.ones(HORIZON)},
            {"action_prior": torch.zeros(HORIZON, 15)},
            {"action_prior": torch.full((HORIZON, 16), float("nan"))},
            {"action_prior": torch.zeros(HORIZON, 16), "action_update_weights": torch.zeros(1, HORIZON)},
            {"action_prior": torch.zeros(HORIZON, 16), "action_update_weights": [-.1] * HORIZON},
            {"action_prior": torch.zeros(HORIZON, 16), "action_update_weights": [1.1] * HORIZON},
            {"action_prior": torch.zeros(HORIZON, 16), "action_update_weights": [float("nan")] * HORIZON},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sample(TinyFastWAM(), **kwargs)


class RTCCodecPolicyTests(unittest.TestCase):
    def test_roundtrip_rebase_and_absolute_grippers_without_clipping(self):
        proc = processor()
        prior = torch.linspace(-2, 3, HORIZON * 16).reshape(HORIZON, 16)
        original = prior.clone()
        old_state = torch.linspace(-.1, .1, 16)
        new_state = old_state + 20
        old_encoded = encode_actions(prior, old_state, proc)
        new_encoded = encode_actions(prior, new_state, proc)
        self.assertGreater(float(new_encoded.abs().max()), 5.)
        self.assertTrue(torch.equal(old_encoded[:, [7, 15]], new_encoded[:, [7, 15]]))
        self.assertFalse(torch.equal(old_encoded[:, ARM_INDICES], new_encoded[:, ARM_INDICES]))
        for state in (old_state, new_state):
            for action in (prior, prior.unsqueeze(0)):
                encoded = encode_actions(action, state, proc)
                torch.testing.assert_close(decode_actions(encoded, state, proc), action, atol=3e-6, rtol=1e-5)
        self.assertTrue(torch.equal(prior, original))
        self.assertLessEqual(float(encode_state(new_state, proc).abs().max()), 5.)
        norm = proc.normalizer.normalizers["action"]["default"]
        self.assertLessEqual(float(norm.forward(torch.full_like(prior, 20.)).abs().max()), 5.)

    def test_policy_frozen_absolute_prior_survives_rebase_and_bf16_sampler(self):
        instance = policy(TinyFastWAM(torch.bfloat16))
        prior = np.linspace(-1.4, 1.1, HORIZON * 16, dtype=np.float32).reshape(HORIZON, 16)
        with patch("scripts.serve_openarm.pack_cameras", return_value=torch.zeros(3, 16, 16)):
            for state in (np.zeros(16, np.float32), np.full(16, 20., np.float32)):
                output = instance.predict({"state": state, "prompt": "pillow"}, action_prior=prior,
                                          action_update_weights=np.zeros(HORIZON, np.float32))
                np.testing.assert_allclose(output["actions"], prior, atol=3e-6, rtol=1e-5)

    def test_policy_off_passes_no_new_kwargs(self):
        model = SimpleNamespace(infer_action=lambda **kw: self.capture(kw))
        self.kwargs = None
        with patch("scripts.serve_openarm.pack_cameras", return_value=torch.zeros(3, 16, 16)):
            policy(model).predict({"state": np.zeros(16, np.float32), "prompt": "pillow"})
        self.assertNotIn("action_prior", self.kwargs)
        self.assertNotIn("action_update_weights", self.kwargs)

    def capture(self, kwargs):
        self.kwargs = kwargs
        return {"action": torch.zeros(HORIZON, 16)}

    def test_policy_rejects_overridden_samplers(self):
        for cls in (FastWAMJoint, FastWAMIDM, FastWAMOptionalIDM):
            model = cls.__new__(cls)
            torch.nn.Module.__init__(model)
            with self.subTest(cls=cls), self.assertRaisesRegex(ValueError, "original FastWAM"):
                policy(model).predict({"state": np.zeros(16), "prompt": "pillow"},
                                      action_prior=np.zeros((HORIZON, 16)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
