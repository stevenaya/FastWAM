"""CPU file-packaging tests; no weights, network, or GPU required."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_openarm_deployment import prepare


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source = self.root / "training"
        source.mkdir()
        for name in ("config.yaml", "dataset_stats.json", "checkpoint.pt", "Wan2.2_VAE.pth", "sample.npz"):
            (source / name).write_bytes(name.encode())
        cache = source / "cache"
        cache.mkdir()
        (cache / "embedding.pt").write_bytes(b"cached prompt")
        self.args = dict(run_dir=source, checkpoint=source / "checkpoint.pt",
                         vae=source / "Wan2.2_VAE.pth", text_cache_dir=cache,
                         warmup_sample=source / "sample.npz", output_dir=self.root / "bundle")

    def test_links_only_large_assets_and_copies_small_assets(self):
        manifest = prepare(**self.args)
        out = self.args["output_dir"]
        self.assertEqual(len(manifest["files"]), 6)
        self.assertTrue((out / "run/weights.pt").is_symlink())
        self.assertTrue((out / "base_models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth").is_symlink())
        self.assertFalse((out / "text_embeds/embedding.pt").is_symlink())
        for record in manifest["files"]:
            if "sha256" in record:
                self.assertEqual(record["sha256"], hashlib.sha256((out / record["path"]).read_bytes()).hexdigest())

    def test_copy_mode_is_portable_and_does_not_overwrite(self):
        prepare(**self.args, copy_weights=True)
        out = self.args["output_dir"]
        self.assertFalse(any(path.is_symlink() for path in out.rglob("*")))
        self.assertEqual((out / "run/weights.pt").read_bytes(), b"checkpoint.pt")
        with self.assertRaises(FileExistsError):
            prepare(**self.args)

    def test_missing_assets_fail_before_creating_output(self):
        self.args["checkpoint"].unlink()
        with self.assertRaises(FileNotFoundError):
            prepare(**self.args)
        self.assertFalse(self.args["output_dir"].exists())


if __name__ == "__main__":
    unittest.main()
