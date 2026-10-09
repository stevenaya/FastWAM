# FastWAM OpenArm Inference Guide

This is the deployment environment for the **original FastWAM** OpenArm model,
inside `openarm-eval-workspace/models/FastWAM`. It does not use the old training
checkout or its Python environment. Keep the eval revision and model submodule
pin together; do not independently update node branches before deployment.

Supported environment target: Linux x86-64, Python 3.10, PyTorch 2.7.1 + CUDA 12.8.
A compatible NVIDIA driver is required for GPU inference. This environment was
built and CPU-tested in the development container; **RTX 5090 memory/latency,
checkpoint RTC inference and robot behavior have not been verified here**.
The CUDA wheels do not install a host driver. No separate `flash-attn` build is
needed: this model uses PyTorch SDPA. Compilation is optional and initially off.

## 1. Build with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) on the target
machine if absent, then open the matching evaluation workspace:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /path/to/openarm-eval-workspace
git submodule update --init models/FastWAM nodes/dora-openarm-local-policy-server
cd models/FastWAM
bash scripts/setup_openarm_inference.sh --group test
source .venv/bin/activate
```

The script runs `uv sync --project deployment --locked --python 3.10`, with
`UV_PROJECT_ENVIRONMENT` defaulting to this model checkout's `.venv`, then checks
dependency compatibility. Downloads/builds/CPU threads are bounded. To use a
different **new** environment path, set `UV_PROJECT_ENVIRONMENT` explicitly and
update the experiment YAML. Do not point it at an active training environment.

`deployment/uv.lock` pins dependencies; its editable model/runtime sources are
relative paths within this eval layout. It uses the official CUDA 12.8 index
explicitly for Torch, torchvision and TorchCodec, following the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
Shared imports currently retain training-compatible packages, including
Accelerate/DeepSpeed. This is not a minimal inference-only dependency set.
`DS_BUILD_OPS=0` avoids building optional DeepSpeed CUDA operators at setup.
The legacy root `pyproject.toml` and training environment are unchanged.

For direct Dora mode, additionally install the optional SDK/caller dependencies:

```bash
bash scripts/setup_openarm_inference.sh --extra dora --group test
```

This extra pins Dora SDK 0.5.0; it must match the deployment graph's Dora runtime.
Socket mode keeps the model and Dora environments separate and is the default.
Setting up the robot driver, UI and dataflow remains the eval workspace's task;
this model environment does not install hardware drivers or start a graph.
Running the setup script again without `--extra dora` removes that optional extra.

Do not use plain root `uv sync` for this deployment profile. The eval YAML uses
`UV_NO_SYNC=1` so launching does not unexpectedly change the provisioned environment.
Rebuild a `.venv` after moving/cloning to another machine; do not copy virtualenvs.
Allow space for CUDA packages/cache in addition to approximately 15 GB of assets.

## 2. Prepare or Transfer Assets

The serving bundle is ignored by Git:

```text
deployment_assets/
  run/config.yaml
  run/dataset_stats.json
  run/weights.pt
  base_models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
  text_embeds/*.pt
  warmup/observation.npz
  manifest.json
```

On a preparation machine with the trained files available, use a new destination:

```bash
python scripts/prepare_openarm_deployment.py \
  --run-dir /path/to/training-run \
  --checkpoint /path/to/training-run/checkpoints/weights/step_100000.pt \
  --vae /path/to/Wan2.2_VAE.pth \
  --text-cache-dir /path/to/text_embeds \
  --warmup-sample /path/to/observation.npz
```

The helper copies config/stats, cached text features and the recorded warmup sample.
It links only the large trained checkpoint (~12 GB) and VAE (~2.8 GB) to resolved
shared-storage paths by default. No dataset videos, optimizer states, original
video DiT weights or T5 encoder weights are needed for cached-prompt serving.
`--copy-weights` creates an all-file bundle instead; `--output-dir` chooses another
new destination. Existing destinations are refused, not overwritten. This helper
targets the saved `redirect_common_files: false` Wan2.2 configuration; converted
VAE formats/model variants need a separate validated preparation path.

**For another machine without the same NAS mounts, materialize the symlinks.**
For example, from the inference machine and model directory:

```bash
rsync -aL --info=progress2 \
  TRAIN_HOST:/path/to/openarm-eval-workspace/models/FastWAM/deployment_assets/ \
  ./deployment_assets/
```

`-L` copies linked weights as files; plain `-a` would leave broken absolute links
unless the same shared storage is mounted. Transfer only to your authorized
machine: the NPZ contains recorded robot observations. The manifest records the
original packaging sources/mode and small-file hashes; it is not a cryptographic
verification of the large weights or a dependency on those source paths.

The training config keeps provenance paths for its original dataset, but serving
does not instantiate that dataset. Explicit asset overrides in the YAML select
the copied text cache and VAE; normalizer statistics are never recomputed.
Text prompts must match an existing cached embedding exactly. New instructions
need compatible T5/tokenizer assets and upstream `scripts/precompute_text_embeds.py`
on a preparation machine; copy the resulting cache entries into this bundle.
The serving process deliberately does not load T5 or silently encode missing text.

## 3. Check Without Moving the Robot

From the model directory with `.venv` activated:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
uv pip check --python .venv/bin/python
python open_eval/policy_server.py --dry-run \
  --run-dir deployment_assets/run \
  --checkpoint-file deployment_assets/run/weights.pt \
  --model-base-path deployment_assets/base_models \
  --text-cache-dir deployment_assets/text_embeds

EVAL_ROOT=$(cd ../.. && pwd)
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$EVAL_ROOT/nodes/dora-openarm-local-policy-server/src" \
  python -m unittest scripts.test_openarm_runtime scripts.test_openarm_rtc_model \
    scripts.test_openarm_deployment -v
```

Dry-run checks sidecars and paths, not checkpoint tensors, GPU execution or robot
safety. CPU tests use mocked model computation. Optional video dataset decoding
through TorchCodec has additional compatible FFmpeg shared-library requirements;
the default server takes decoded RGB/NPZ inputs and does not decode video files.

On an explicitly assigned idle GPU, run a real checkpoint/socket smoke with a
new output directory. This starts only a private policy socket, never Dora or a robot:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_openarm_runtime.py \
  --run-dir deployment_assets/run \
  --checkpoint deployment_assets/run/weights.pt \
  --sample deployment_assets/warmup/observation.npz \
  --model-base-path deployment_assets/base_models \
  --text-cache-dir deployment_assets/text_embeds \
  --output-dir artifacts/deployment-smoke
```

Add `--rtc` and use a different output directory for the executor-prior smoke.
It checks a simulated plan's frozen prefix; it is not a physical execution test.
Keep `--compile-action-infer` off initially and measure warmed latency/peak memory
on the actual inference GPU. A CUDA-version check alone does not prove 5090 readiness.

## 4. Start an Evaluation Experiment

The included `exp/FastWAM/fastwam_pillow_100k.yaml` resolves environment/assets from
`${script_dir}/models/FastWAM`, not an old sibling training checkout. Review output
directory, socket, GPU, prompt and camera wiring. It is an **experimental RTC**
example: set `rtc-source: none` for ordinary inference first. RTC requires executor
feedback and does not guarantee task success or safety. Current model output is
32 absolute 16D actions at 30 Hz; inference Hz is only a request-rate cap.

```bash
cd ../..
./launch_inference.sh --dry-run exp/FastWAM/fastwam_pillow_100k.yaml
# Only after offline validation and explicit hardware authorization:
./launch_inference.sh exp/FastWAM/fastwam_pillow_100k.yaml
```

Keep the model `.venv` active for the launcher's Python/PyYAML and uv commands.
The selected default dataflow controls physical hardware. Direct Dora execution,
hardware behavior, CUDA compilation and this new deployment's GPU RTC remain
separate validation items, not consequences of a passing environment installation.
