# Fast-WAM on OpenArm Pillow

Official repository: https://github.com/yuantianyuan01/FastWAM

Baseline revision: `7faa71108368fbb3b6885649f112af607427a2d4`.
This integration uses the original Fast-WAM variant, not Optional IDM. Both
video and action experts are trained. Deployment uses the action-only path;
it does not generate future video. A smoke test verifies execution, not robot
task success. Do offline evaluation before connecting trained outputs to hardware.

## Storage and data

- Project and isolated Python 3.10 environment: `/workspace/FastWAM` and `.venv`.
- All model weights and training runs: `/mnt/syno127/volume1/stevenaya/fast_wam`.
- `checkpoints` in the project is only a symlink to that storage.
- Original dataset: `/mnt/syno127/volume1/openarm-dataset/N17/converted/pillow-0702-filtered`.
- Original modality: `/mnt/syno127/volume1/stevenaya/groot/N17/modality.json`.
- Source selection: GR00T menu `pillow_0702`, dataset `pillow_selected0702_final`.
- Private dataset: `data/pillow_0702`. Metadata and Parquet are separate copies.
  Videos are read through a symlink; nothing writes into the original videos.
- `preparation.json` records source metadata checksums and a seed-42 episode split:
  2297 training and 121 validation episodes. Normalization uses training episodes only.

Camera baseline: `head_left`, `wrist_left`, `wrist_right`. The other head camera
and ceiling camera are not used. The OpenArm cam-v2 crops are retained, followed
by Fast-WAM resizing and the RoboTwin-style 384x320 three-camera mosaic.
Training uses 9 video frames at offsets 0,4,...,32 and 32 actions at 30 Hz.
Only current state and current images are provided for deployment.

State/action order is unchanged from the dataset:
`right_arm[7], right_gripper, left_arm[7], left_gripper`.
Joint actions are relative to the current state; gripper targets remain absolute.
Normalization is upstream z-score with its [-5,5] clipping, not GR00T percentile
normalization. Deployment returns absolute actions in original dataset units.

## Environment and preparation

```bash
cd /workspace/FastWAM
source scripts/openarm_env.sh
taskset -c "$CPU_SET" python scripts/prepare_openarm_models.py
taskset -c "$CPU_SET" python scripts/preprocess_action_dit_backbone.py \
  --model-config "$DIFFSYNTH_MODEL_BASE_PATH/openarm_model.yaml" \
  --output "$DIFFSYNTH_MODEL_BASE_PATH/openarm_action_dit.pt" \
  --device cuda --dtype bfloat16
taskset -c "$CPU_SET" python scripts/precompute_text_embeds.py task=openarm_pillow_100k
taskset -c "$CPU_SET" python scripts/check_openarm_data.py
```

Wan2.2-TI2V-5B and UMT5 weights are reused read-only from the existing DreamZero
cache. The action backbone is initialized by the upstream interpolation script;
its action input/output layers and the 16D state projection start fresh.
The dataset is already prepared. `scripts/prepare_openarm.py` only accepts a new
output directory to avoid overwriting an existing preparation.

## Training

The task config is `configs/task/openarm_pillow_100k.yaml`. GPU visibility defaults
to 0,1,2,3; CPU affinity is limited to logical CPUs 0-31, with 4 loader workers per
rank and bounded BLAS/OpenMP threads. Override `CPU_SET` for another allocation.
The current upstream tensor core requires gradient checkpointing disabled.

The formal run uses 4 GPUs, batch 4 per GPU and accumulation 2 (global batch 32),
100,000 optimizer steps, AdamW LR 1e-4, 5% warmup, cosine decay and BF16 ZeRO-2.

```bash
cd /workspace/FastWAM
bash scripts/start_openarm_tmux.sh
tmux attach -t fastwam_pillow_100k
```

Detach with Ctrl-b then d. This survives terminal/SSH disconnection, but not the
container/Slurm allocation ending. The launcher prints the run directory. It
contains `train.log`, resolved `config.yaml`, `dataset_stats.json`, offline W&B
logs, inference previews and checkpoints. `exit_code` appears when training exits.

Weight-only checkpoints: `checkpoints/weights/step_010000.pt` and every 10k steps.
Full resumable checkpoints: `checkpoints/state/step_010000/` (optimizer and sampler).

```bash
OUTPUT_DIR=/mnt/syno127/volume1/stevenaya/fast_wam/runs/RECOVERY_RUN \
  bash scripts/train_openarm.sh \
  resume=/mnt/syno127/volume1/stevenaya/fast_wam/runs/RUN/checkpoints/state/step_010000
```

`max_steps=100000` means total optimizer steps, including restored steps.
Hydra overrides such as `batch_size=2` are accepted by both launchers.

## Deployment smoke test

Use a separate GPU allocation; do not compete with the running four-card job.

```bash
source scripts/openarm_env.sh
CUDA_VISIBLE_DEVICES=0 taskset -c "$CPU_SET" python scripts/serve_openarm.py \
  --run-dir /mnt/syno127/volume1/stevenaya/fast_wam/runs/RUN \
  --checkpoint /mnt/syno127/volume1/stevenaya/fast_wam/runs/RUN/checkpoints/weights/step_010000.pt
```

The server binds localhost:18010. `GET /health` reports the loaded step and action
order. `POST /infer` takes an NPZ body (`allow_pickle=False`) with `state` float32
shape [16], `prompt` a scalar string, and the three camera keys as uint8 RGB HWC
arrays at original resolution. It returns JSON with absolute `actions` [32,16],
`fps`, `checkpoint_step`, and `latency_ms`. There is no actuator connection.

```bash
python scripts/test_openarm_server.py
```

This sends the real observation prepared by the data check twice and saves the
responses in `artifacts/openarm_checks/server_checks.json`. Known Pillow prompts
use the cached T5 embeddings. New prompts must be cached with the upstream text
precompute script (`+override_instruction='...'`) before deployment.

## Scoped changes

OpenArm data/serving helpers and configs are separate from the original model.
One upstream processor shape assertion converts both sides to tuples; comparing
`torch.Size` to a Python list otherwise rejects equal dimensions.
Throughput logging includes gradient accumulation, the final checkpoint is not
saved twice when it coincides with the save interval, and normal training exit
closes Accelerate trackers/process groups.
Data checks cover episode split, source metadata, first-frame pixel equality,
state normalization, finite tensors, action shapes and unclipped action roundtrip.

## Verified on four A100 80GB GPUs

- Real data tests: matching train/deploy pixels and state, max unclipped action
  roundtrip error 4.47e-8, disjoint episode splits, unchanged source metadata.
- Three optimizer steps completed; action/video losses were both finite. Step 3
  total loss 2.2831. Full states and weight-only files saved successfully.
- Saved step-2 checkpoint loaded strictly by the HTTP server. Two real observation
  requests returned finite [32,16] actions. Uncompiled 10-step latency was about
  908 ms cold / 505 ms warm. These are smoke timings, not a production benchmark.
- Full step-3 optimizer/scheduler/sampler/RNG state restored on four GPUs; two
  updates with batch 4 and accumulation 2 succeeded. Peak allocated GPU memory
  was about 45.05 GiB. This batch probe changes batch geometry and is not an
  exact-data-order continuation; the formal run starts fresh.
- Records: `/mnt/syno127/volume1/stevenaya/fast_wam/validation` and
  `runs/pillow_resume_probe_b4/resume_check.json` under the same storage root.
