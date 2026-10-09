"""Four-GPU optimizer-resume/batch probe; deliberately skips duplicate saves."""

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from fastwam.runtime import build_datasets, _resolve_train_device
from fastwam.trainer import Wan22Trainer
from fastwam.utils import misc
from fastwam.utils.logging_config import setup_logging


class ProbeTrainer(Wan22Trainer):
    def save_checkpoint(self):
        return {"weights_path": "disabled_for_probe", "state_path": "disabled_for_probe"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    args = parser.parse_args()
    saved = json.loads((args.resume / "trainer_state.json").read_text())
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            "task=openarm_pillow_100k", f"output_dir={args.output}", f"resume={args.resume}",
            f"max_steps={saved['global_step'] + 2}", f"batch_size={args.batch_size}",
            f"gradient_accumulation_steps={args.grad_accum}", "save_every=0", "eval_every=0",
            "log_every=1", "wandb.enabled=false", "model.skip_dit_load_from_pretrain=true",
            "model.action_dit_pretrained_path=null",
        ])
    setup_logging()
    misc.register_work_dir(args.output)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=_resolve_train_device())
    train, val = build_datasets(cfg.data)
    trainer = ProbeTrainer(model, train, val, cfg=cfg)
    assert trainer.global_step == saved["global_step"]
    assert trainer.accelerator.gradient_accumulation_steps == args.grad_accum
    trainer.train()
    peaks = trainer.accelerator.gather(torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=trainer.accelerator.device))
    if trainer.accelerator.is_main_process:
        report = {"restored_step": saved["global_step"], "final_step": trainer.global_step,
                  "world_size": trainer.accelerator.num_processes, "batch_size_per_gpu": args.batch_size,
                  "gradient_accumulation": args.grad_accum, "peak_memory_gib": peaks.tolist()}
        (args.output / "resume_check.json").write_text(json.dumps(report, indent=2) + "\n")
        OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
        print(json.dumps(report, indent=2), flush=True)
    trainer.accelerator.wait_for_everyone()
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()
