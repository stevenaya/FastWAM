"""Reuse cached official Wan weights without changing their source files."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    root = Path(__file__).resolve().parents[1]
    source = Path("/mnt/syno127/volume1/stevenaya/dreamzero/checkpoints")
    cache = Path("/mnt/syno127/volume1/stevenaya/fast_wam/checkpoints")
    cache.mkdir(parents=True, exist_ok=True)
    if not (root / "checkpoints").exists():
        (root / "checkpoints").symlink_to(cache, target_is_directory=True)
    links = {
        cache / "Wan-AI/Wan2.2-TI2V-5B": source / "Wan2.2-TI2V-5B",
        cache / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl": source / "umt5-xxl",
    }
    for dest, src in links.items():
        if not src.is_dir():
            raise FileNotFoundError(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.symlink_to(src, target_is_directory=True)
        print(f"{dest} -> {dest.resolve()}")
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=["task=openarm_pillow_100k"])
    OmegaConf.save(cfg.model, cache / "openarm_model.yaml", resolve=True)


if __name__ == "__main__":
    main()
