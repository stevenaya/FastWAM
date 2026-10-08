"""Serve trained FastWAM through OpenArm's socket/SHM or direct Dora runtime."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

# Support the workspace launcher's `python open_eval/policy_server.py` invocation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from openarm_policy_runtime import Backend, ModelSession, Prediction, serve
from open_eval.rtc import ExecutionRTC


def inspect_run(run_dir, checkpoint, text_cache_dir=None):
    """Read sidecars only; --dry-run must not load Torch or checkpoint tensors."""
    import yaml

    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    json.loads((run_dir / "dataset_stats.json").read_text())
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    train = config["data"]["train"]
    processor = train["processor"]
    cameras = processor["shape_meta"]["images"]
    if [m["key"] for m in cameras] != ["head_left", "wrist_left", "wrist_right"]:
        raise ValueError("Expected head_left/wrist_left/wrist_right in the trained mosaic order")
    if processor["action_output_dim"] != 16 or processor["proprio_output_dim"] != 16:
        raise ValueError("This OpenArm backend requires 16D joint action and state")
    cache = Path(text_cache_dir or train["text_embedding_cache_dir"])
    if not cache.is_dir():
        raise FileNotFoundError(f"Missing text embedding cache: {cache}; set --text-cache-dir")
    return {"run_dir": str(run_dir), "checkpoint": str(checkpoint),
            "camera_keys": [m["key"] for m in cameras], "action_horizon": train["num_frames"] - 1,
            "text_cache_dir": str(cache), "fps": 30}


class FastWAMBackend(Backend):
    name = "fastwam"

    def __init__(self, policy, *, warmup_sample=None, warmup_steps=0, rtc=None):
        self.policy = policy
        self.camera_shapes = {
            "camera_" + meta["key"]: tuple(meta["raw_shape"][1:]) + (3,)
            for meta in policy.processor.shape_meta["images"]
        }
        self.camera_fields = tuple(self.camera_shapes)
        self.horizon = int(policy.cfg.data.train.num_frames) - 1
        self.warmup_sample, self.warmup_steps = warmup_sample, warmup_steps
        self.rtc = rtc

    def warmup(self):
        if not self.warmup_steps:
            return
        with np.load(self.warmup_sample, allow_pickle=False) as sample:
            observation = {name: sample[name].copy() for name in sample.files}
        timings = []
        for index in range(self.warmup_steps):
            started = time.perf_counter_ns()
            self.policy.predict(observation)  # Discard synthetic episode output before readiness.
            if index >= max(1, self.warmup_steps // 2):
                timings.append(time.perf_counter_ns() - started)
        if self.rtc is not None and timings:
            self.rtc.delay_ns = float(np.mean(timings))
            self.rtc.reset()

    def predict(self, observation):
        started_ns = time.time_ns()
        index = observation.current_index()
        state = observation.qpos[index]
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError("OpenArm requires 16 finite state values, right 7+1 then left 7+1")
        model_obs = {"state": state, "prompt": observation.prompt}
        for field, shape in self.camera_shapes.items():
            camera = observation.cameras[field][index]
            if camera.shape != shape or camera.dtype != np.uint8:
                raise ValueError(f"{field}: expected original-resolution RGB uint8 {shape}, got {camera.shape}")
            model_obs[field.removeprefix("camera_")] = camera
        steps = observation.metadata.get("denoising_steps")
        if steps is not None:
            steps = int(steps)
            if steps <= 0:
                raise ValueError("denoising_steps must be positive")
            self.policy.steps = steps
        execution = None
        rtc_inputs = {}
        if self.rtc is not None:
            prepared = self.rtc.prepare(observation, started_ns)
            if prepared is None:
                if self.rtc.probe_due:
                    probe_started = time.perf_counter_ns()
                    self.policy.predict(model_obs)
                    self.rtc.delay_ns = float(time.perf_counter_ns() - probe_started)
                return Prediction(None, log_data={"rtc": {
                    "skipped": "no_safe_window", "latency_probe": self.rtc.probe_due,
                }})
            prior, weights, execution = prepared
            if prior is not None:
                rtc_inputs = dict(action_prior=prior, action_update_weights=weights)
        result = self.policy.predict(model_obs, **rtc_inputs)
        positions = np.asarray(result["actions"], dtype=np.float32)
        if positions.shape != (self.horizon, 16) or not np.isfinite(positions).all():
            raise ValueError(f"Expected finite absolute actions [{self.horizon},16], got {positions.shape}")
        # decode_actions already undoes z-score normalization and rebases arm joints.
        return Prediction(
            positions, interval_ns=int(1e9 / result["fps"]), cutoff_hz=15,
            execution=execution,
            timing={"fastwam_predict_ms": result["latency_ms"]},
            log_data={"checkpoint_step": result["checkpoint_step"],
                      "denoising_steps": self.policy.steps,
                      "rtc": {"enabled": self.rtc is not None, "used": bool(rtc_inputs),
                              "frozen_steps": execution["action_window_start"] if execution else 0}},
        )

    def reset(self, reason):
        # Text embeddings are immutable; only executor priors belong to an episode.
        if self.rtc is not None:
            self.rtc.reset()
        return False

    def close(self):
        self.policy = None


def create_backend(args):
    # Model/environment stay private to this process; no dependency installation.
    if args.model_base_path is not None:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path)
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    import torch
    from scripts.serve_openarm import OpenArmPolicy

    torch.set_num_threads(args.torch_num_threads)
    policy = OpenArmPolicy(
        args.run_dir, args.checkpoint_file, steps=args.denoising_steps, device=args.device,
        text_cache_dir=args.text_cache_dir, compile_action_infer=args.compile_action_infer,
        seed=args.seed,
    )
    logging.info("Loaded FastWAM step %s on %s", policy.checkpoint_step, args.device)
    rtc = ExecutionRTC(
        int(policy.cfg.data.train.num_frames) - 1, margin_ms=args.rtc_margin_ms,
        ramp_steps=args.rtc_ramp_steps, ramp_rate=args.rtc_ramp_rate,
        max_lateness_ms=args.rtc_max_lateness_ms,
    ) if args.rtc_source == "executor" else None
    return FastWAMBackend(policy, warmup_sample=args.warmup_sample,
                          warmup_steps=args.warmup_steps, rtc=rtc)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-file", "--checkpoint", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path)
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transport", choices=("socket", "dora"), default="socket")
    parser.add_argument("--socket-path", default="/dev/shm/fastwam-policy.socket")
    parser.add_argument("--local-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    parser.add_argument("--compile-action-infer", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-sample", type=Path, help="Recorded RGB/state/prompt NPZ, never executed")
    parser.add_argument("--rtc-source", choices=("none", "executor"), default="none")
    parser.add_argument("--rtc-margin-ms", type=float, default=10)
    parser.add_argument("--rtc-ramp-steps", type=int, default=6)
    parser.add_argument("--rtc-ramp-rate", type=float, default=5)
    parser.add_argument("--rtc-max-lateness-ms", type=float, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.denoising_steps <= 0 or args.torch_num_threads <= 0:
        parser.error("denoising-steps and torch-num-threads must be positive")
    if args.warmup_steps < 0 or (args.warmup_steps and args.warmup_sample is None):
        parser.error("positive warmup-steps requires --warmup-sample")
    if args.rtc_ramp_steps < 0 or any(not math.isfinite(value) or value < 0 for value in (
        args.rtc_margin_ms, args.rtc_ramp_rate, args.rtc_max_lateness_ms,
    )):
        parser.error("RTC margin, ramp and lateness must be finite and nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    contract = inspect_run(args.run_dir, args.checkpoint_file, args.text_cache_dir)
    logging.info("Inference contract: %s", json.dumps(contract))
    if args.dry_run:
        print(json.dumps({**contract, "transport": args.transport, "device": args.device}, indent=2))
        return
    if args.transport == "dora":
        from dora_openarm_local_policy_server.dora_runner import serve_dora
        serve_dora(lambda: create_backend(args), prompt=args.prompt)
    else:
        backend = create_backend(args)
        try:
            backend.warmup()
            serve(args.socket_path, lambda: ModelSession(backend), listen=args.local_server)
        finally:
            backend.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("FastWAM server stopped")
