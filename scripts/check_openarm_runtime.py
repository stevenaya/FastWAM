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
import pyarrow as pa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-base-path", type=Path, required=True)
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
        arrow = temp / "obs.arrow"
        arrays = [pa.array([array.reshape(-1).tolist()],
                          type=pa.list_(pa.float32() if name == "position" else pa.uint8()))
                  for name, array in fields.items()]
        batch = pa.record_batch(arrays + [pa.array([prompt])], names=list(fields) + ["task_prompt"])
        with pa.OSFile(str(arrow), "wb") as sink, pa.ipc.new_file(sink, batch.schema) as writer:
            writer.write_batch(batch)
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
                   "--socket-path", address, "--local-server", "--seed", "42", "--infer-hz", "1000",
                   "--action-window-start", "3", "--action-window-size", "8",
                   "--warmup-steps", "2", "--warmup-sample", str(args.sample),
                   "--chunk-log-path", str(args.output_dir / "chunks.jsonl")]
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
                        arrow_response = request({"data_path": str(arrow), "metadata": metadata, "reset": True})
                        ring_response = request({"transport": "shm_ring_v1", "shm": descriptor,
                                                 "metadata": metadata, "reset": True})
                        shifted = request({"transport": "shm_ring_v1", "shm": {**descriptor, "sequence": 2},
                                           "metadata": metadata})
                        changed_steps = request({"data_path": str(arrow),
                                                 "metadata": {**metadata, "denoising_steps": 4}})
                        assert ring_response["input_sequence"] == 1
                        assert arrow_response["metadata"]["reset"] and ring_response["metadata"]["reset"]
                        assert not shifted["metadata"]["reset"] and shifted["metadata"]["action_window_start"] == 3
                        values = np.asarray(ring_response["positions"])
                        assert values.shape == (8, 16) and np.isfinite(values).all()
                        difference = float(np.max(np.abs(values - np.asarray(arrow_response["positions"]))))
                        assert difference < 1e-5, difference
                        report = {"checkpoint": str(args.checkpoint), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                                  "action_shape": list(values.shape), "arrow_shm_max_abs_diff": difference,
                                  "arrow_policy_ms": arrow_response["timing"]["policy_ms"],
                                  "shm_policy_ms": ring_response["timing"]["policy_ms"],
                                  "four_step_policy_ms": changed_steps["timing"]["policy_ms"],
                                  "ping": True, "shm_ack": True, "reset_and_window": True,
                                  "robot_connected": False}
                        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(report, indent=2), flush=True)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()


if __name__ == "__main__":
    main()
