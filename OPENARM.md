# Fast-WAM on OpenArm Pillow

For a fresh inference machine, follow [the uv deployment guide](deployment/README.md).
It builds this checkout's `.venv` and prepares `deployment_assets/`; it does not
require the legacy `/workspace/FastWAM` training checkout.

Fork: https://github.com/stevenaya/FastWAM, branch `openarm-pillow-runtime`.
The 100k run completed with exit code 0 on 2026-09-13. The original training
checkout and weights remain unchanged; this fork includes its source changes
and the shared OpenArm policy-runtime adapter added on 2026-10-02.

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

## Shared OpenArm Policy Runtime

`open_eval/policy_server.py` reuses `scripts.serve_openarm.OpenArmPolicy`, including
the training camera transforms, cached T5 prompt format, state normalization and
absolute action decoding. No HTTP hop is used in the evaluation workspace.
The backend returns the full `[32,16]` action sequence through `ModelSession`.
The evaluation caller owns pacing, windows, reset boundaries, chunk IDs and logs.
Socket inputs use `shm_ring_v1` only; direct Dora uses Arrow through the same caller
session. The old `Session`/Arrow-file socket API is not supported. Immutable text
embeddings are retained on reset; executor plans are cleared.

Runtime contract: `reazon-research/openarm-eval-workspace` commit
`e6b2801` and its pinned node/model submodules. Use that set together, not arbitrary
node branch tips. This wrapper follows the same model-only boundary as its
GR00T and OpenPI entrypoints.

From the evaluation workspace, use `demo_gr00t/fastwam_pillow_100k.yaml` with
`launch_inference.sh --dry-run` first. The example now uses this submodule's `.venv`
and `deployment_assets/`, provisioned by the uv deployment guide. `UV_NO_SYNC=1`
prevents launch-time changes to that environment. Large linked weights still need
the declared shared-storage mount or materialization when transferring machines.

Standalone invocation (set these paths for the target machine):

```bash
export PYTHONPATH=/path/to/openarm-eval-workspace/packages/openarm-policy-runtime/src:$PWD/src
export DIFFSYNTH_SKIP_DOWNLOAD=true
python open_eval/policy_server.py \
  --run-dir /path/to/training-run \
  --checkpoint-file /path/to/step_100000.pt \
  --model-base-path /path/to/base-model-cache \
  --text-cache-dir /path/to/text_embeds \
  --local-server --socket-path /dev/shm/fastwam-policy.socket \
  --denoising-steps 10
```

Required deployment assets are the weight-only checkpoint, the run's resolved
`config.yaml` and `dataset_stats.json`, the referenced Wan VAE/base-model cache,
and cached T5 embeddings for the instructions used. Full optimizer states and
the training dataset are not used by this server. `--model-base-path` and
`--text-cache-dir` relocate assets without modifying saved configs.
Keep source RGB resolutions: head left `720x1280`, wrists `600x960`; the backend
rejects missing/wrong-size views instead of silently changing the crop geometry.

New instructions need a matching cached T5 embedding; missing prompts fail
explicitly. Prepare them with the existing `scripts/precompute_text_embeds.py`
and `+override_instruction='...'` before serving. `--compile-action-infer` opts
into the upstream action-inference compiler; it is off in the validated baseline.
Metadata `denoising_steps` changes the step count for subsequent calls.

`--transport dora` uses `dora_openarm_local_policy_server.dora_runner`. The model environment
must separately provide a Dora version compatible with the workspace; default
socket mode does not import Dora. Do not merge Torch/JAX/model environments.
Optional `--warmup-steps N --warmup-sample observation.npz` runs recorded inputs
before publishing readiness and discards their actions. Warmup does not consume
the first real action/reset marker. `--seed` fixes inference noise for comparisons.

## Executor-aligned RTC

Set `policy.args.rtc-source: executor` in the example YAML; use `none` for native
ordinary inference. The dataflow must connect `actions-executor/execution_plan`
to `policy-server/execution_plan`. It already does so at the pinned eval revision.
Feedback describes the adopted pre-filter command plan, not measured motor motion.
The caller handles the single pending handoff and stale attempt rejection.

At observation time `t0`, inference start `ts`, action interval `dt`, measured
latency `D` and margin `M`, freeze `F = ceil((max(0, ts-t0) + D + M)/dt)` steps.
Warmup initializes latency; executor receipt feedback updates an EWMA once per
sample. If `F >= H`, return a prefill/skip instead of an unusable timed chunk.
For fresh observations, a discarded inference probe at most once every five
seconds can replace a stale oversized latency estimate; it never emits actions.
Align the absolute plan to `t0 + i*dt`; a just-preceding observation can use one
retained previous plan. Completed plans hold their last target, as in the executor.
The backend returns timing and `based_on_chunk_id` in `Prediction.execution`.
Bootstrap starts at zero; subsequent caller windows begin at `F`.

The absolute prior is rebased to the current raw state through training's relative
arm transform; grippers stay absolute. Apply the checkpoint's affine action
normalization without its training-only `[-5,5]` clip: clipping a rebased prior
would change frozen absolute targets. Normal training/state processing is unchanged.
This inverse codec is not a robot safety clamp; actuator limits remain separate.

In the original FastWAM action sampler, sigma decreases from one to zero and
velocity is `noise - clean_action`. With the same initial Gaussian noise `eps`:

```text
anchor_next = (1 - sigma_next) * normalized_prior + sigma_next * eps
proposal = native_scheduler.step(velocity, native_delta, latent)
latent_next = (1 - update_weight) * anchor_next + update_weight * proposal
```

Every denoising step applies the constraint, not just the final decoded result.
Weights are zero in the frozen prefix, start at zero at takeover and ramp to the
free region's weight one. `rtc-ramp-rate: 0` is linear; positive rates use a
normalized exponential. Frozen endpoints are retained in FP32. This is inference
time inpainting, not training-time RTC or proof of the paper's guided-RTC quality.
The current adapter supports the original FastWAM only, not overridden Joint/IDM
samplers. Ordinary no-prior sampling and its RNG sequence are unchanged.

`rtc-margin-ms`, `rtc-ramp-steps`, `rtc-ramp-rate`, `rtc-max-lateness-ms` configure
the model algorithm. `infer-hz`, `action-window-start/size`, reset-gap and chunk
logging remain caller settings in the YAML, not server CLI arguments. Resets and
disconnect discard the model's plan cache; clock alignment assumes a shared host.

## Validation

CPU tests (runtime and local policy node source directories on PYTHONPATH):

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest scripts.test_openarm_runtime scripts.test_openarm_rtc_model -v
```

`scripts/check_openarm_runtime.py` launches a temporary socket server, uses a
recorded NPZ observation, checks full SHM predictions, input release and reset,
then stops the server. Add `--rtc` to simulate executor feedback and check the
decoded frozen prefix against its absolute prior. Use a new output directory and
an explicitly assigned free GPU; it never
launches Dora or a robot driver. Its outputs contain local observation/action
records and should not be committed. `artifacts/` is ignored for this reason.

Historical 2026-10-02 smoke, before the ModelSession migration: step 100k loaded
strictly on A100 80GB GPU 0; Arrow/SHM max action
difference was 0 with seed 42. After two warmups, 10-step policy calls took
approximately 508-519 ms and one 4-step call 246 ms. These are a few smoke samples,
not p95/throughput or task-success benchmarks. Playback is 30 Hz; `infer-hz=10`
is only a rate cap and does not make inference run at 10 Hz. Full robot evaluation
and direct Dora execution with this backend remain untested.

2026-10-09 migration: 22 CPU tests pass (10 protocol/timeline, 12 sampler/codec),
including rebase outside the clipping range, absolute grippers, frozen/ramp/free
regions, nonlinear schedules and unchanged ordinary sampling/RNG. The compiled
interface test uses a mock, not GPU compilation. All eight local GPUs were busy;
no GPU checkpoint, RTC latency or physical-robot validation was run for this revision.

Training scripts now locate the repo relatively. Set `OPENARM_DATA_ROOT`,
`DIFFSYNTH_MODEL_BASE_PATH`, `UV_PROJECT_ENVIRONMENT` and `OUTPUT_DIR` as needed;
`OPENARM_BASE_MODEL_SOURCE` relocates the optional DreamZero cache import.
The lock snapshot retains the trained package versions; its editable source is
now `-e .`, not the old `/workspace/FastWAM` checkout. CUDA-specific packages need
the PyTorch CUDA 12.8 wheel index when provisioning a fresh environment.
