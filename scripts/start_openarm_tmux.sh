#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
SESSION="${SESSION:-fastwam_pillow_100k}"
RUN_NAME="${RUN_NAME:-pillow_0702_100k_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/syno127/volume1/stevenaya/fast_wam/runs/$RUN_NAME}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR"
git rev-parse HEAD > "$OUTPUT_DIR/upstream_commit.txt"
git diff > "$OUTPUT_DIR/local_changes.patch"
tar -czf "$OUTPUT_DIR/openarm_integration.tar.gz" \
  OPENARM.md requirements-openarm.lock.txt src/fastwam/openarm.py \
  configs/data/openarm_pillow.yaml configs/task/openarm_pillow_100k.yaml \
  scripts/*openarm*.py scripts/*openarm*.sh
printf -v command '%q ' env "OUTPUT_DIR=$OUTPUT_DIR" bash scripts/run_logged_openarm.sh "$@"
tmux new-session -d -s "$SESSION" -c "$PWD" "$command"
tmux set-option -t "$SESSION" remain-on-exit on
printf 'Session: %s\nOutput: %s\nLog: %s/train.log\n' "$SESSION" "$OUTPUT_DIR" "$OUTPUT_DIR"
