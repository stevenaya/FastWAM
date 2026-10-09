"""Prepare a small serving bundle; never copy the training dataset or environment."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(run_dir, checkpoint, vae, text_cache_dir, warmup_sample, output_dir, *, copy_weights=False):
    files = {
        "run/config.yaml": Path(run_dir) / "config.yaml",
        "run/dataset_stats.json": Path(run_dir) / "dataset_stats.json",
        "run/weights.pt": Path(checkpoint),
        "base_models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth": Path(vae),
        "warmup/observation.npz": Path(warmup_sample),
    }
    cached = sorted(Path(text_cache_dir).glob("*.pt"))
    if not cached:
        raise ValueError("No cached T5 embeddings found")
    files.update({"text_embeds/" + path.name: path for path in cached})
    files = {name: path.resolve(strict=True) for name, path in files.items()}
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"Expected a file: {path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    large = {"run/weights.pt", "base_models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"}
    records = []
    for name, source in files.items():
        target = output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        linked = name in large and not copy_weights
        if linked:
            target.symlink_to(source)
        else:
            shutil.copyfile(source, target)
        record = dict(path=name, source=str(source), bytes=source.stat().st_size,
                      mode="symlink" if linked else "copy")
        if name not in large:
            record["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        records.append(record)
    manifest = dict(schema_version=1, copy_weights=copy_weights, files=records)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True, help="Official Wan2.2_VAE.pth for redirect_common_files=false")
    parser.add_argument("--text-cache-dir", type=Path, required=True)
    parser.add_argument("--warmup-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "deployment_assets")
    parser.add_argument("--copy-weights", action="store_true", help="Copy large weights instead of linking shared storage")
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
