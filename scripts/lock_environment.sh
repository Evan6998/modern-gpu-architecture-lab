#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results/locked-environment}"
mkdir -p "$OUT"
python -m pip freeze > "$OUT/pip-freeze.txt"
python -m pip check > "$OUT/pip-check.txt"
python -m mgpu.doctor --output "$OUT/environment.json"
echo "Recorded current environment in $OUT (not a claim of cross-machine compatibility)."
