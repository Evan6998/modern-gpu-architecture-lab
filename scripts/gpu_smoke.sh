#!/usr/bin/env bash
# First-run check for a GPU machine, run from the repository root:
#   bash scripts/gpu_smoke.sh
# It covers the toolchain, the harness, the starter extension builds, the
# library baselines and profiler access. Every section runs even if an earlier
# one fails; the summary at the end lists what did not pass. It checks the
# environment, not any student kernel: a starter that fails to BUILD is an
# environment problem, a starter that stops at its TODO is the expected state.
set -uo pipefail
mkdir -p results
FAILED=()
section() { printf '\n===== %s =====\n' "$*"; }
check() {  # check <label> <command...>
  local label="$1"; shift
  if "$@"; then echo "[ok] $label"; else echo "[FAILED] $label"; FAILED+=("$label"); fi
}

section "GPU and driver"
check "nvidia-smi" nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv

section "Toolchain"
python --version
check "torch sees CUDA" python -c "
import torch
print('torch', torch.__version__, '| cuda runtime', torch.version.cuda, '| SM', torch.cuda.get_device_capability())
print('arch list', torch.cuda.get_arch_list())"
check "nvcc" bash -c 'nvcc --version | tail -2'
g++ --version | head -1
for tool in ninja ncu nsys compute-sanitizer cuobjdump; do
  printf '%-18s %s\n' "$tool" "$(command -v "$tool" || echo MISSING)"
done

section "Harness on a GPU machine"
check "make doctor" bash -c 'make doctor > /dev/null && python -c "
import json; d = json.load(open(\"results/environment.json\"))
print(d[\"devices\"]); print(\"packages\", d[\"packages\"])"'
check "make check" make check
check "make cpu" make cpu

starter_builds() {  # the starter must COMPILE and then stop at its TODO
  local a="$1" variant="$2" todo="$3" log="results/$1-starter.log"
  MGPU_BUILD_VERBOSE=1 python -m mgpu.grade "$a" --variant "$variant" > "$log" 2>&1
  if grep -q "$todo" "$log"; then
    echo "built, then failed at $todo as intended"
    grep -m3 "ptxas info" "$log" || true
    return 0
  fi
  echo "did not reach $todo; last lines of $log:"; tail -25 "$log"; return 1
}
section "Starter extensions compile"
check "A1 starter builds" starter_builds a1 naive "TODO(A1)"
check "A2 starter builds" starter_builds a2 simt "TODO(A2)"

section "Library baselines pass the benchmark gate"
check "a1 transpose" python -m mgpu.bench a1 --op transpose --impl library --suite full
check "a1 softmax" python -m mgpu.bench a1 --op softmax --impl library --suite full
check "a2 gemm" python -m mgpu.bench a2 --op gemm --impl library --suite full
check "a5 attention" python -m mgpu.bench a5 --op attention --impl library --suite full
check "eviction buffer not counted as workspace" bash -c '
  python -m mgpu.bench a1 --op transpose --impl library --suite smoke --evict-mb 64 --output results/evict-check &&
  python -c "
import csv; row = list(csv.DictReader(open(\"results/evict-check/bench.csv\")))[-1]
print(\"peak_extra_allocated_bytes =\", row[\"peak_extra_allocated_bytes\"], \"| cache_policy =\", row[\"cache_policy\"])
assert int(row[\"peak_extra_allocated_bytes\"]) < 64 * 1024 * 1024"'

ncu_reads_counters() {
  command -v ncu > /dev/null || { echo "ncu is not installed"; return 1; }
  local cmd=(python -m mgpu.bench a1 --op transpose --impl library --suite smoke --warmup 1 --repeats 1 --output results/ncu-check)
  if ncu --set default -o results/ncu-probe -f "${cmd[@]}" > results/ncu-probe.log 2>&1; then
    echo "ncu works without sudo"; return 0
  fi
  grep -m2 -iE "ERR_NVGPUCTRPERM|permission|error" results/ncu-probe.log || tail -5 results/ncu-probe.log
  echo "retrying with sudo"
  if sudo env "PATH=$PATH" "$(command -v ncu)" --set default -o results/ncu-probe -f "${cmd[@]}" > results/ncu-probe-sudo.log 2>&1; then
    echo "ncu works with sudo"; return 0
  fi
  tail -8 results/ncu-probe-sudo.log; return 1
}
section "Nsight Compute can read GPU performance counters"
check "ncu" ncu_reads_counters

section "Summary"
if [ ${#FAILED[@]} -eq 0 ]; then echo "all checks passed"; else printf 'failed: %s\n' "${FAILED[@]}"; exit 1; fi
