"""CPU tests for the FastWAM shared-runtime backend; no model or robot startup."""

import io
import json
import contextlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openarm_policy_runtime import Observation, Session, serve_connection
from open_eval.policy_server import FastWAMBackend, parse_args


class FakePolicy:
    def __init__(self):
        self.processor = SimpleNamespace(shape_meta={"images": [
            {"key": key, "raw_shape": [3, 2, 3]}
            for key in ("head_left", "wrist_left", "wrist_right")
        ]})
        self.cfg = SimpleNamespace(data=SimpleNamespace(train=SimpleNamespace(num_frames=33)))
        self.steps = 10
        self.observations = []

    def predict(self, obs):
        self.observations.append(obs)
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
            timestamp=123, qpos=np.stack([np.full(16, 0.1), np.full(16, 0.2)]).astype(np.float32),
            cameras={name: np.full((2, 2, 3, 3), i + 1, np.uint8)
                     for i, name in enumerate(self.backend.camera_fields)},
            prompt="task", delta_indices=(0, -32), metadata={"inference_trial_id": "one"},
        )
        self.session = Session(self.backend, infer_hz=1e9, action_window_start=3,
                               action_window_size=5, timing_log_every=0)
        self.addCleanup(self.session.close)

    def test_mapping_current_frame_absolute_actions_and_steps(self):
        self.obs.metadata["denoising_steps"] = 4
        result = self.backend.predict(self.obs)
        self.assertEqual(self.policy.steps, 4)
        np.testing.assert_array_equal(self.policy.observations[0]["state"], self.obs.qpos[0])
        self.assertEqual(list(self.policy.observations[0]), ["state", "prompt", "head_left", "wrist_left", "wrist_right"])
        self.assertEqual(result.positions.shape, (32, 16))
        np.testing.assert_allclose(result.positions[0], 0.1)
        self.assertEqual(result.interval_ns, 33333333)

    def test_reset_window_and_prompt_boundaries(self):
        first = self.session.handle_observation(self.obs)
        second = self.session.handle_observation(self.obs)
        self.assertTrue(first["metadata"]["reset"])
        self.assertEqual(first["metadata"]["action_window_start"], 0)
        self.assertFalse(second["metadata"]["reset"])
        self.assertEqual(second["metadata"]["action_window_start"], 3)
        self.obs.prompt = "new prompt"
        self.assertFalse(self.session.handle_observation(self.obs)["metadata"]["reset"])
        self.obs.metadata["inference_trial_id"] = "two"
        trial = self.session.handle_observation(self.obs)
        self.assertTrue(trial["metadata"]["reset"])
        np.testing.assert_allclose(trial["positions"][0], 0.1)

    def test_bad_camera_missing_camera_state_and_steps_fail(self):
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

    def test_warmup_does_not_consume_execution_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            sample = Path(temp) / "warmup.npz"
            np.savez(sample, state=self.obs.qpos[0], prompt="task")
            backend = FastWAMBackend(self.policy, warmup_steps=2, warmup_sample=sample)
            backend.warmup()
        self.assertEqual(len(self.policy.observations), 2)
        self.assertTrue(self.session.handle_observation(self.obs)["metadata"]["reset"])

    def test_arrow_and_shm_use_identical_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            fields = {"position": self.obs.qpos, **self.obs.cameras}
            metadata = {"timestamp": 123, "history_delta_indices": "0,-32"}
            for name in self.backend.camera_fields:
                metadata.update({f"{name}.height": 2, f"{name}.width": 3})
            arrays = [pa.array(array.reshape(len(array), -1).tolist(),
                               type=pa.list_(pa.float32() if key == "position" else pa.uint8()))
                      for key, array in fields.items()]
            batch = pa.record_batch(arrays + [pa.array(["task", "task"])], names=list(fields) + ["task_prompt"])
            arrow_path = temp / "sample.arrow"
            with pa.OSFile(str(arrow_path), "wb") as sink, pa.ipc.new_file(sink, batch.schema) as writer:
                writer.write_batch(batch)
            payload, descriptors = bytearray(), {}
            for name, array in fields.items():
                descriptors[name] = {"offset": len(payload), "nbytes": array.nbytes,
                                     "dtype": array.dtype.str, "shape": list(array.shape)}
                payload.extend(array.tobytes())
            ring_path = temp / "ring"
            ring_path.write_bytes(payload)
            shm = {"path": str(ring_path), "ring_size": len(payload), "slot_count": 1,
                   "slot_size": len(payload), "slot": 0, "payload_size": len(payload),
                   "sequence": 8, "fields": descriptors, "task_prompt": "task"}
            requests = [
                {"ping": True},
                {"reset": True, "data_path": str(arrow_path), "metadata": metadata},
                {"reset": True, "transport": "shm_ring_v1", "shm": shm, "metadata": metadata},
            ]
            stream = Duplex(requests)
            serve_connection(stream, self.session)
            self.assertTrue(stream.responses[0]["ready"])
            self.assertEqual(stream.responses[2]["input_sequence"], 8)
            self.assertEqual(stream.responses[1]["positions"], stream.responses[2]["positions"])
            self.assertTrue(stream.responses[2]["metadata"]["reset"])

    def test_cli_rejects_invalid_options(self):
        for extra in (["--denoising-steps", "0"], ["--warmup-steps", "1"],
                      ["--action-window-size", "0"], ["--infer-hz", "nan"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--run-dir", "/tmp/run", "--checkpoint", "/tmp/step.pt", *extra])


if __name__ == "__main__":
    unittest.main()
