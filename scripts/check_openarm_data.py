"""Check real validation frames, action roundtrip and protected metadata."""

import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
import torch

from fastwam.openarm import decode_actions, encode_state, pack_cameras
from fastwam.utils import misc


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/openarm_checks"
    output.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(output)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["task=openarm_pillow_100k"])
    dataset = instantiate(cfg.data.val)
    loader = dataset.lerobot_dataset
    processor = loader.processor
    manifest = json.loads((Path(cfg.data.train.dataset_dirs[0]) / "preparation.json").read_text())
    episodes = loader.multi_dataset._datasets[0].episodes
    assert set(episodes) == set(manifest["val_episodes"])
    assert not set(episodes) & set(manifest["train_episodes"])
    for path, digest in manifest["protected_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    roundtrip_errors = []
    for idx in (0, min(100, len(dataset) - 1), len(dataset) - 1):
        raw = loader.multi_dataset[idx]
        sample = dataset[idx]
        assert sample["video"].shape == (3, 9, 384, 320)
        assert sample["action"].shape == (32, 16)
        assert sample["context"].shape == (128, 4096)
        state = raw["observation.state"][0]
        obs = {"state": state.numpy(), "prompt": raw["task"]}
        for meta in processor.shape_meta["images"]:
            key = meta["key"]
            frame = (raw[f"observation.images.{key}"][0] * 255).to(torch.uint8)
            obs[key] = frame.permute(1, 2, 0).numpy()
        torch.testing.assert_close(pack_cameras(obs, processor)[0], sample["video"][:, 0])
        torch.testing.assert_close(encode_state(state, processor)[0], sample["proprio"][0])
        decoded = decode_actions(sample["action"], state, processor)
        valid = (~sample["action_is_pad"]).unsqueeze(-1) & (sample["action"].abs() < 4.999)
        error = (decoded - raw["action"]).abs()[valid].max().item()
        assert error < 1e-5, error
        roundtrip_errors.append(error)
        assert all(torch.isfinite(x).all() for x in sample.values() if torch.is_tensor(x))
        if idx == 0:
            np.savez(output / "observation.npz", **obs)
            torch.save(sample, output / "validation_sample.pt")
    report = {"validation_frames": len(dataset), "train_episodes": len(manifest["train_episodes"]),
              "val_episodes": len(episodes), "action_roundtrip_max_abs": max(roundtrip_errors),
              "train_deploy_pixels_match": True, "source_metadata_unchanged": True}
    (output / "data_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
