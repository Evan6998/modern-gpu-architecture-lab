"""Collect reproducibility metadata without changing drivers, clocks or settings."""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from mgpu.hardware import compatible


def command(args):
    if not shutil.which(args[0]):
        return None
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {"exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def snapshot():
    packages = {}
    for name in ("torch", "pytest", "triton", "nvidia-cutlass-dsl", "ninja"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version,
        "packages": packages, "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(), "devices": [],
        "nvcc": command(["nvcc", "--version"]),
        "nvidia_smi": command(["nvidia-smi"]),
        "topology": command(["nvidia-smi", "topo", "-m"]),
        "git": command(["git", "rev-parse", "HEAD"]),
        "git_dirty": command(["git", "status", "--porcelain"]),
        "environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST",
            "NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA")},
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cc = (p.major, p.minor)
            result["devices"].append({
                "index": i, "name": p.name, "compute_capability": list(cc),
                "total_memory": p.total_memory,
                "multiprocessor_count": p.multi_processor_count,
                "lab_families": [s for s in ("cuda", "ampere", "hopper", "blackwell") if compatible(s, cc)],
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(snapshot(), ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
