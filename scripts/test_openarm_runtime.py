"""CPU contract tests against the current OpenArm runtime and caller session."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_policy_runtime import ModelSession, Observation, serve_connection
from dora_openarm_local_policy_server.session import Session
from open_eval.policy_server import FastWAMBackend, parse_args
from open_eval.rtc import ExecutionRTC


class FakePolicy:
    def __init__(self):
        self.processor = SimpleNamespace(shape_meta={"images": [
            {"key": key, "raw_shape": [3, 2, 3]}
            for key in ("head_left", "wrist_left", "wrist_right")
        ]})
        self.cfg = SimpleNamespace(data=SimpleNamespace(train=SimpleNamespace(num_frames=33)))
        self.steps = 10
        self.observations, self.priors = [], []

    def predict(self, obs, **rtc):
        self.observations.append(obs)
        self.priors.append(rtc)
        actions = obs["state"] + np.arange(32, dtype=np.float32)[:, None] * 0.001
        return {"actions": actions.tolist(), "fps": 30, "checkpoint_step": 100000, "latency_ms": 1.0}


class Duplex(io.StringIO):
    def __init__(self, requests):
        super().__init__("".join(json.dumps(request) + "\n" for request in requests))
        self.responses = []

    def write(self, value):
        self.responses.append(json.loads(value))


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.backend = FastWAMBackend(self.policy)
        self.obs = Observation(
            timestamp=1_000_000_000,
            qpos=np.stack([np.full(16, 0.1), np.full(16, 0.2)]).astype(np.float32),
            cameras={name: np.full((2, 2, 3, 3), i + 1, np.uint8)
                     for i, name in enumerate(self.backend.camera_fields)},
            prompt="task", delta_indices=(0, -32),
            metadata={"timestamp": 1_000_000_000, "inference_trial_id": "one"},
        )
        self.session = ModelSession(self.backend)
        self.addCleanup(self.session.close)

    def plan(self, chunk="old", start=None):
        return dict(chunk_id=chunk, start_timestamp_ns=self.obs.timestamp if start is None else start,
                    interval_ns=ExecutionRTC.interval_ns,
                    positions=np.repeat(np.arange(32)[:, None], 16, axis=1).tolist())

    def test_mapping_full_absolute_actions_and_steps(self):
        self.obs.metadata["denoising_steps"] = 4
        result = self.session.handle_observation(self.obs)
        self.assertEqual(self.policy.steps, 4)
        np.testing.assert_array_equal(self.policy.observations[0]["state"], self.obs.qpos[0])
        self.assertEqual(list(self.policy.observations[0]), ["state", "prompt", "head_left", "wrist_left", "wrist_right"])
        self.assertEqual(result["positions"].shape, (32, 16))
        self.assertNotIn("metadata", result)
        self.assertEqual(result["interval"], 33333333)
        self.assertEqual(self.policy.priors, [{}])

    def test_caller_alone_owns_windows_and_reset(self):
        caller = Session(action_window_start=3, action_window_size=5, timing_log_every=0)
        self.addCleanup(caller.close)
        for index, start in enumerate((0, 3)):
            request = caller.prepare(1, self.obs.metadata, self.obs.prompt)
            full = self.session.handle_observation(self.obs, control=request)
            result = caller.complete(request, full)
            self.assertEqual(full["positions"].shape, (32, 16))
            self.assertEqual(result["metadata"]["action_window_start"], start)
            self.assertEqual(result["metadata"]["reset"], index == 0)
            np.testing.assert_array_equal(result["positions"], full["positions"][start:start + 5])
            caller.sent(result)
        request = caller.prepare(2, self.obs.metadata, self.obs.prompt)
        self.assertTrue(request["restart_execution"])

    def test_invalid_inputs_fail_before_model(self):
        with self.assertRaises(KeyError):
            self.backend.predict(Observation(1, self.obs.qpos, {}, "task"))
        self.obs.cameras["camera_head_left"] = np.zeros((2, 4, 4, 3), np.uint8)
        with self.assertRaises(ValueError):
            self.backend.predict(self.obs)
        self.assertEqual(self.policy.observations, [])
        self.obs.cameras["camera_head_left"] = np.zeros((2, 2, 3, 3), np.uint8)
        self.obs.qpos[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.backend.predict(self.obs)
        self.obs.qpos[0, 0] = 0
        self.obs.metadata["denoising_steps"] = 0
        with self.assertRaises(ValueError):
            self.backend.predict(self.obs)

    def test_shm_matches_direct_observation_and_releases_slot(self):
        expected = self.backend.predict(self.obs).positions
        with tempfile.TemporaryDirectory() as temp:
            fields = {"position": self.obs.qpos, **self.obs.cameras}
            payload, descriptors = bytearray(), {}
            for name, array in fields.items():
                descriptors[name] = {"offset": len(payload), "nbytes": array.nbytes,
                                     "dtype": array.dtype.str, "shape": list(array.shape)}
                payload.extend(array.tobytes())
            ring = Path(temp) / "ring"
            ring.write_bytes(payload)
            shm = {"path": str(ring), "ring_size": len(payload), "slot_count": 1,
                   "slot_size": len(payload), "slot": 0, "payload_size": len(payload),
                   "sequence": 8, "fields": descriptors, "task_prompt": "task"}
            request = {"transport": "shm_ring_v1", "shm": shm, "reset_reason": "request",
                       "restart_execution": True, "metadata": {
                           **self.obs.metadata, "history_delta_indices": "0,-32"}}
            stream = Duplex([{"ping": True}, request])
            serve_connection(stream, self.session)
            self.assertTrue(stream.responses[0]["ready"])
            result = stream.responses[1]
            self.assertEqual(result["input_sequence"], 8)
            self.assertEqual(result["released_input_sequences"], [8])
            self.assertTrue(result["reset_applied"])
            np.testing.assert_array_equal(result["positions"], expected)

    def test_rtc_bootstrap_freeze_and_zero_start_ramp(self):
        rtc = self.backend.rtc = ExecutionRTC(32, margin_ms=0, ramp_steps=3, ramp_rate=0)
        with patch("open_eval.policy_server.time.time_ns", return_value=self.obs.timestamp):
            first = self.backend.predict(self.obs)
            self.assertEqual(first.execution["based_on_chunk_id"], "")
            self.assertEqual(self.policy.priors[-1], {})
            self.obs.execution_plan = self.plan()
            self.obs.metadata["rtc_frozen_steps"] = 2
            result = self.backend.predict(self.obs)
        self.assertEqual(result.execution["action_window_start"], 2)
        self.assertEqual(result.execution["takeover_timestamp_ns"], self.obs.timestamp + 2 * rtc.interval_ns)
        passed = self.policy.priors[-1]
        np.testing.assert_array_equal(passed["action_prior"], self.plan()["positions"])
        np.testing.assert_allclose(passed["action_update_weights"][:6], [0, 0, 0, 1/3, 2/3, 1])

    def test_rtc_latency_updated_once_and_saturated_delay_skips(self):
        rtc = self.backend.rtc = ExecutionRTC(32, margin_ms=0)
        self.obs.execution_plan = {**self.plan(), "sample_chunk_id": "sample",
                                   "inference_started_timestamp_ns": 100,
                                   "received_timestamp_ns": 100 + rtc.interval_ns * 3}
        for _ in range(2):
            prepared = rtc.prepare(self.obs, self.obs.timestamp)
            self.assertEqual(prepared[2]["action_window_start"], 3)
        self.assertEqual(rtc.delay_ns, rtc.interval_ns * 3)
        with patch("open_eval.policy_server.time.time_ns", return_value=self.obs.timestamp + 32 * rtc.interval_ns):
            self.assertIsNone(self.backend.predict(self.obs).positions)
        self.assertEqual(self.policy.observations, [])

    def test_rtc_previous_plan_and_reset(self):
        rtc = self.backend.rtc = ExecutionRTC(32, margin_ms=0)
        self.obs.execution_plan = self.plan()
        rtc.prepare(self.obs, self.obs.timestamp)
        self.obs.execution_plan = self.plan("new", self.obs.timestamp + rtc.interval_ns)
        prior, _, _ = rtc.prepare(self.obs, self.obs.timestamp)
        np.testing.assert_array_equal(prior[:2, 0], [0, 0])
        for reason in ("request", "trial", "prompt", "idle", "disconnect"):
            self.backend.reset(reason)
            self.assertIsNone(rtc.plan)
            self.assertIsNone(rtc.previous_plan)
        self.assertIsNone(rtc.prepare(self.obs, self.obs.timestamp))
        self.obs.metadata["execution_restart"] = True
        self.assertIsNone(rtc.prepare(self.obs, self.obs.timestamp)[0])

    def test_saturated_latency_probe_recovers_without_emitting_actions(self):
        rtc = self.backend.rtc = ExecutionRTC(32, margin_ms=0)
        rtc.delay_ns = 40 * rtc.interval_ns
        self.obs.execution_plan = self.plan()
        with patch("open_eval.policy_server.time.time_ns", return_value=self.obs.timestamp):
            probe = self.backend.predict(self.obs)
            self.assertIsNone(probe.positions)
            self.assertTrue(probe.log_data["rtc"]["latency_probe"])
            self.assertEqual(len(self.policy.observations), 1)
            self.assertIsNotNone(self.backend.predict(self.obs).positions)
            rtc.delay_ns = 40 * rtc.interval_ns
            skipped = self.backend.predict(self.obs)
            self.assertIsNone(skipped.positions)
            self.assertFalse(skipped.log_data["rtc"]["latency_probe"])
            self.assertEqual(len(self.policy.observations), 2)

    def test_warmup_has_no_execution_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.npz"
            np.savez(path, state=self.obs.qpos[0], prompt="task")
            rtc = ExecutionRTC(32)
            backend = FastWAMBackend(self.policy, warmup_steps=2, warmup_sample=path, rtc=rtc)
            backend.warmup()
        self.assertGreater(rtc.delay_ns, 0)
        self.assertIsNone(rtc.plan)
        self.assertEqual(len(self.policy.observations), 2)

    def test_cli_rejects_invalid_options_and_old_session_flags(self):
        for extra in (["--denoising-steps", "0"], ["--warmup-steps", "1"],
                      ["--rtc-ramp-steps", "-1"], ["--rtc-margin-ms", "nan"],
                      ["--action-window-size", "0"], ["--arrow-memory-map"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--run-dir", "/tmp/run", "--checkpoint", "/tmp/step.pt", *extra])


if __name__ == "__main__":
    unittest.main()
