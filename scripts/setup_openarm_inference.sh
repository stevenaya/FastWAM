#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$ROOT/.venv}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-2}
export UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export DS_BUILD_OPS=0 PYTHONDONTWRITEBYTECODE=1
# The serving environment is independent of any existing training checkout/env.
uv sync --project "$ROOT/deployment" --python 3.10 --locked "$@"
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
printf 'Inference Python: %s/bin/python\n' "$UV_PROJECT_ENVIRONMENT"
