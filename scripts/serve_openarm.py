"""Local action-only HTTP deployment. Requests are RGB/state/prompt NPZ files."""

import argparse
import hashlib
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import time

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.openarm import decode_actions, encode_actions, encode_state, pack_cameras


class OpenArmPolicy:
    def __init__(self, run_dir, checkpoint, steps=10, device="cuda", *,
                 text_cache_dir=None, compile_action_infer=False, seed=None):
        self.cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
        self.processor = instantiate(self.cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(Path(run_dir) / "dataset_stats.json")
        )
        self.model = instantiate(
            self.cfg.model, device=device, model_dtype=torch.bfloat16,
            skip_dit_load_from_pretrain=True, action_dit_pretrained_path=None,
            load_text_encoder=False,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.mot.load_state_dict(payload["mot"], strict=True)
        self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.model.eval().requires_grad_(False)
        self.checkpoint_step = payload["step"]
        self.steps = steps
        self.text_cache_dir = Path(text_cache_dir or self.cfg.data.train.text_embedding_cache_dir)
        self.compile_action_infer = compile_action_infer
        self.seed = seed
        self.contexts = {}

    @torch.inference_mode()
    def predict(self, observation, *, action_prior=None, action_update_weights=None):
        """Predict absolute actions, optionally inpainting an absolute [H,16] RTC prior.

        Weights [H] mean 0=frozen and 1=free; omitted weights freeze the full prior.
        Only the original FastWAM sampler supports this contract.
        """
        started = time.perf_counter()
        state = np.asarray(observation["state"], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("state must be 16 finite values in dataset order")
        rtc_kwargs = {}
        if action_prior is None and action_update_weights is not None:
            raise ValueError("action_update_weights requires action_prior")
        if action_prior is not None:
            from fastwam.models.wan22.fastwam import FastWAM

            if type(self.model).infer_action is not FastWAM.infer_action:
                raise ValueError("Action-prior RTC supports the original FastWAM sampler, not overridden Joint/IDM samplers")
            prior = torch.as_tensor(action_prior, dtype=torch.float32)
            horizon = self.cfg.data.train.num_frames - 1
            if prior.shape != (horizon, 16) or not torch.isfinite(prior).all():
                raise ValueError(f"action_prior must be finite [{horizon},16] absolute actions")
            rtc_kwargs = {"action_prior": encode_actions(prior, state, self.processor),
                          "action_update_weights": action_update_weights}
        prompt = DEFAULT_PROMPT.format(task=str(np.asarray(observation["prompt"]).item()))
        if prompt not in self.contexts:
            hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            cache = self.text_cache_dir
            cache /= f"{hashed}.t5_len{self.cfg.data.train.context_len}.wan22ti2v5b.pt"
            if not cache.is_file():
                raise FileNotFoundError(
                    f"Prompt has no cached T5 embedding: {cache}. Use scripts/precompute_text_embeds.py "
                    "to cache this exact instruction, or pass --text-cache-dir."
                )
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            context = payload["context"].clone()
            context[~payload["mask"].bool()] = 0
            self.contexts[prompt] = (context.unsqueeze(0), torch.ones_like(payload["mask"], dtype=torch.bool).unsqueeze(0))
        context, mask = self.contexts[prompt]
        result = self.model.infer_action(
            prompt=None, input_image=pack_cameras(observation, self.processor),
            proprio=encode_state(state, self.processor),
            context=context, context_mask=mask,
            action_horizon=self.cfg.data.train.num_frames - 1,
            num_inference_steps=self.steps, text_cfg_scale=1.0,
            seed=self.seed, compile_action_infer=self.compile_action_infer,
            **rtc_kwargs,
        )
        actions = decode_actions(result["action"], state, self.processor).numpy()
        if not np.isfinite(actions).all():
            raise RuntimeError("Model returned non-finite actions")
        return {"actions": actions.tolist(), "fps": 30, "checkpoint_step": self.checkpoint_step,
                "latency_ms": (time.perf_counter() - started) * 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18010)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    policy = OpenArmPolicy(args.run_dir, args.checkpoint, args.steps)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            data = json.dumps(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": "Not found"})
            self.reply(200, {"ready": True, "checkpoint_step": policy.checkpoint_step,
                             "cameras": [m["key"] for m in policy.processor.shape_meta["images"]],
                             "action_order": "right_arm[7],right_gripper,left_arm[7],left_gripper"})

        def do_POST(self):
            if self.path != "/infer":
                return self.reply(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 32 * 1024 * 1024:
                    raise ValueError("Expected an NPZ body of at most 32 MiB")
                with np.load(io.BytesIO(self.rfile.read(length)), allow_pickle=False) as obs:
                    result = policy.predict(obs, action_prior=obs.get("action_prior"),
                                            action_update_weights=obs.get("action_update_weights"))
            except (ValueError, KeyError, FileNotFoundError) as exc:
                return self.reply(400, {"error": str(exc)})
            self.reply(200, result)

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Ready: http://{args.host}:{args.port} (checkpoint step {policy.checkpoint_step})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
