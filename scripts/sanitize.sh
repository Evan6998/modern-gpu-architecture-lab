#!/usr/bin/env bash
set -euo pipefail
A="${1:-a1}"
case "$A" in
 a1) DIR=a1_memory;; a2) DIR=a2_ampere;; a3) DIR=a3_hopper;;
 a4) DIR=a4_blackwell;; a5) DIR=a5_attention;;
 *) echo 'Use a1..a5; profile A6 under torchrun separately.' >&2; exit 2;;
esac
command -v compute-sanitizer >/dev/null || { echo 'compute-sanitizer is not installed' >&2; exit 2; }
for TOOL in memcheck racecheck synccheck; do
 compute-sanitizer --tool "$TOOL" --error-exitcode 1    python -m pytest "assignments/$DIR" -m gpu --run-gpu --require-gpu --strict-hardware -x -q
done
