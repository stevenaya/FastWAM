"""Offline GPU/socket smoke test with a recorded sample. No Dora or robot control."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    parser.add_argument("--rtc", action="store_true", help="Also verify a simulated executor handoff")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    with np.load(args.sample, allow_pickle=False) as sample:
        fields = {"position": sample["state"].astype(np.float32)[None]}
        prompt = str(sample["prompt"].item())
        for key in ("head_left", "wrist_left", "wrist_right"):
            fields["camera_" + key] = sample[key][None].copy()
    metadata = {"timestamp": time.time_ns(), "history_delta_indices": "0", "inference_trial_id": "offline-smoke"}
    for name, value in fields.items():
        if name.startswith("camera_"):
            metadata.update({name + ".height": value.shape[1], name + ".width": value.shape[2]})
    with tempfile.TemporaryDirectory(prefix="fastwam-socket-") as temp:
        temp = Path(temp)
        payload, descriptors = bytearray(), {}
        for name, array in fields.items():
            descriptors[name] = {"offset": len(payload), "nbytes": array.nbytes,
                                 "dtype": array.dtype.str, "shape": list(array.shape)}
            payload.extend(array.tobytes())
        ring = temp / "ring"
        ring.write_bytes(payload)
        descriptor = {"path": str(ring), "ring_size": len(payload), "slot_count": 1,
                      "slot_size": len(payload), "slot": 0, "payload_size": len(payload),
                      "sequence": 1, "fields": descriptors, "task_prompt": prompt}
        address = str(temp / "policy.socket")
        command = [sys.executable, str(root / "open_eval/policy_server.py"),
                   "--run-dir", str(args.run_dir), "--checkpoint", str(args.checkpoint),
                   "--model-base-path", str(args.model_base_path), "--text-cache-dir", str(args.text_cache_dir),
                   "--socket-path", address, "--local-server", "--seed", "42",
                   "--warmup-steps", "4", "--warmup-sample", str(args.sample)]
        if args.rtc:
            command += ["--rtc-source", "executor"]
        with (args.output_dir / "server.log").open("w") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 600
                while not Path(address).exists():
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited {process.returncode}; see {args.output_dir / 'server.log'}")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server did not become ready within 600s")
                    time.sleep(0.5)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.settimeout(120)
                    client.connect(address)
                    with client.makefile("rw") as io:
                        def request(value):
                            io.write(json.dumps(value) + "\n")
                            io.flush()
                            response = json.loads(io.readline())
                            if "error" in response:
                                raise RuntimeError(response["error"])
                            return response

                        assert request({"ping": True})["ready"]
                        responses = []
                        def infer(sequence, **overrides):
                            result = request({"transport": "shm_ring_v1",
                                              "shm": {**descriptor, "sequence": sequence},
                                              "metadata": {**metadata, "timestamp": time.time_ns()},
                                              "chunk_id": str(sequence), **overrides})
                            assert result["input_sequence"] == sequence
                            assert sequence in result["released_input_sequences"]
                            values = np.asarray(result["positions"])
                            assert values.shape == (32, 16) and np.isfinite(values).all()
                            responses.append(result)
                            return result

                        first = infer(1, reset_reason="request", restart_execution=True)
                        repeat = infer(2, reset_reason="request", restart_execution=True)
                        assert first["reset_applied"] and repeat["reset_applied"]
                        difference = float(np.max(np.abs(np.asarray(first["positions"]) - repeat["positions"])))
                        rtc_error = None
                        if args.rtc:
                            origin = time.time_ns()
                            plan = dict(chunk_id="adopted-2", sample_chunk_id="2",
                                        start_timestamp_ns=origin, interval_ns=repeat["interval"],
                                        positions=repeat["positions"],
                                        inference_started_timestamp_ns=repeat["execution"]["inference_started_timestamp_ns"],
                                        received_timestamp_ns=origin)
                            timed = infer(3, execution_plan=plan, metadata={**metadata, "timestamp": origin})
                            execution = timed["execution"]
                            frozen = execution["action_window_start"]
                            assert 0 < frozen < 32 and execution["based_on_chunk_id"] == "adopted-2"
                            # Same origin and action interval: the frozen prefix must decode back to the plan.
                            rtc_error = float(np.max(np.abs(np.asarray(timed["positions"])[:frozen] -
                                                           np.asarray(plan["positions"])[:frozen])))
                            assert rtc_error < 1e-5, rtc_error
                        reset = infer(4, reset_reason="trial", restart_execution=True)
                        if args.rtc:
                            assert reset["execution"]["based_on_chunk_id"] == ""
                        report = {"checkpoint": str(args.checkpoint), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                                  "action_shape": [32, 16], "repeat_max_abs_diff": difference,
                                  "policy_ms": [r["timing"]["policy_ms"] for r in responses],
                                  "rtc": args.rtc, "rtc_frozen_max_abs_diff": rtc_error,
                                  "ping": True, "shm_release": True, "reset": True,
                                  "robot_connected": False}
                        (args.output_dir / "responses.json").write_text(json.dumps(responses) + "\n")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            if process.returncode not in (0, -signal.SIGINT):
                raise RuntimeError(f"Server shutdown failed with exit code {process.returncode}")
            report.update(passed=True, server_exit_code=process.returncode)
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
