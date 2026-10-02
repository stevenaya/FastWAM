"""Create an independent LeRobot metadata/Parquet copy and train-only statistics."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Moments:
    def __init__(self, dim):
        self.n = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.square = self.sum.copy()
        self.minimum = np.full(dim, np.inf)
        self.maximum = np.full(dim, -np.inf)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return
        if not np.isfinite(values).all():
            raise ValueError("Non-finite values in source state/action")
        self.n += len(values)
        self.sum += values.sum(0)
        self.square += np.square(values).sum(0)
        self.minimum = np.minimum(self.minimum, values.min(0))
        self.maximum = np.maximum(self.maximum, values.max(0))

    def result(self):
        mean = self.sum / self.n
        std = np.sqrt(np.maximum(self.square / self.n - mean**2, 0))
        return {f"global_{k}": v.tolist() for k, v in {
            "mean": mean, "std": std, "min": self.minimum, "max": self.maximum
        }.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu", default="/workspace/Openarm-GR00T/train_on_dgx/datasets/openarm_n17.yaml")
    parser.add_argument("--dataset", default="pillow_selected0702_final")
    parser.add_argument("--output", type=Path, default=Path(os.environ.get(
        "OPENARM_DATA_ROOT", Path(__file__).resolve().parents[1] / "data/pillow_0702"
    )))
    parser.add_argument("--user", default="stevenaya")
    args = parser.parse_args()
    menu = yaml.safe_load(Path(args.menu).read_text())
    entry = menu["datasets"][args.dataset]
    expand = lambda s: os.path.expandvars(str(s).replace("${DGX_USER}", args.user))
    source = (Path(expand(menu["dataset_root"])) / expand(entry["folder"])).resolve()
    modality_path = Path(expand(entry["modality_json_path"])).resolve()
    output = args.output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the source dataset")
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    protected = {str(p): sha256(p) for p in (source / "meta").glob("*.json*")}
    protected[str(modality_path)] = sha256(modality_path)
    modality = json.loads(modality_path.read_text())
    relative = []
    for key in ("right_arm", "left_arm"):
        a, s = modality["action"][key], modality["state"][key]
        if (a["start"], a["end"]) != (s["start"], s["end"]):
            raise ValueError(f"State/action slice mismatch for {key}")
        relative.extend(range(a["start"], a["end"]))
    expected = list(range(7)) + list(range(8, 15))
    if relative != expected:
        raise ValueError(f"Update OpenArm config for this joint layout: {relative}")
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    shutil.copyfile(modality_path, output / "meta" / "modality.json")
    shutil.copytree(source / "data", output / "data")
    (output / "videos").symlink_to(source / "videos", target_is_directory=True)
    info_path = output / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    cameras = {f"observation.images.{k}" for k in ("head_left", "wrist_left", "wrist_right")}
    info["features"] = {k: v for k, v in info["features"].items() if v["dtype"] != "video" or k in cameras}
    info["total_videos"] = info["total_episodes"] * len(cameras)
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    ids = np.arange(info["total_episodes"])
    np.random.default_rng(42).shuffle(ids)
    split = int(len(ids) * 0.95)
    train_ids, val_ids = sorted(ids[:split].tolist()), sorted(ids[split:].tolist())
    stats = {"action": Moments(16), "state": Moments(16)}
    for episode in tqdm(train_ids, desc="Train-only statistics"):
        relpath = info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        table = pq.read_table(output / relpath, columns=["action", "observation.state"])
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        stats["state"].add(state)
        # Only valid targets contribute, matching the action padding loss mask.
        for offset in range(min(32, len(action))):
            target = action[offset:].copy()
            target[:, relative] -= state[:len(target), relative]
            stats["action"].add(target)
    payload = {key: {"default": value.result()} for key, value in stats.items()}
    (output / "dataset_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    for path, digest in protected.items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"Source metadata changed during preparation: {path}")
    manifest = {"source": str(source), "modality_source": str(modality_path),
                "protected_sha256": protected, "train_episodes": train_ids, "val_episodes": val_ids,
                "seed": 42, "action_horizon": 32, "relative_indices": relative,
                "cameras": sorted(cameras), "original_data_modified": False}
    (output / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {output}: {len(train_ids)} train / {len(val_ids)} val episodes")


if __name__ == "__main__":
    main()
