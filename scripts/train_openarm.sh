#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FastWAM
source scripts/openarm_env.sh
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/syno127/volume1/stevenaya/fast_wam/runs/pillow_0702_100k}"
echo "Fast-WAM: GPUs=$CUDA_VISIBLE_DEVICES CPUs=$CPU_SET data=$OPENARM_DATA_ROOT output=$OUTPUT_DIR"
exec taskset -c "$CPU_SET" accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 4 --main_process_port "${MASTER_PORT:-29619}" scripts/train.py \
  task=openarm_pillow_100k "output_dir=$OUTPUT_DIR" "$@"
