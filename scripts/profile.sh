#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "Usage: $0 ncu|nsys a1 [benchmark options...]" >&2; exit 2; fi
TOOL="$1"; A="$2"; shift 2
mkdir -p "results/$A"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
case "$TOOL" in
  nsys)
    command -v nsys >/dev/null || { echo 'Nsight Systems is not installed' >&2; exit 2; }
    nsys profile --trace=cuda,nvtx,osrt -o "results/$A/nsys-$STAMP"       python -m mgpu.bench "$A" --impl student --suite smoke --warmup 1 --repeats 1 "$@"
    ;;
  ncu)
    command -v ncu >/dev/null || { echo 'Nsight Compute is not installed' >&2; exit 2; }
    FILTER=()
    if [[ -n "${NCU_KERNEL_REGEX:-}" ]]; then FILTER=(--kernel-name "regex:$NCU_KERNEL_REGEX"); fi
    ncu --set full --target-processes all "${FILTER[@]}" -o "results/$A/ncu-$STAMP"       python -m mgpu.bench "$A" --impl student --suite smoke --warmup 1 --repeats 1 "$@"
    ;;
  *) echo 'Choose ncu or nsys' >&2; exit 2 ;;
esac
