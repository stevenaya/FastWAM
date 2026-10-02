#!/usr/bin/env bash
set -uo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
bash scripts/train_openarm.sh "$@" 2>&1 | tee "$OUTPUT_DIR/train.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$OUTPUT_DIR/exit_code"
exit "$status"
