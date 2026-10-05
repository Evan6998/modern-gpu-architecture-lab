"""Lazy C++/CUDA build: importing the lab never invokes nvcc."""
from functools import lru_cache
from pathlib import Path
import os
import shutil


@lru_cache(maxsize=None)
def load_extension(assignment: str):
    import torch
    from torch.utils.cpp_extension import load, CUDA_HOME
    from mgpu.registry import ASSIGNMENTS
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA extension requires an NVIDIA GPU")
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin/nvcc").exists():
        raise RuntimeError("CUDA toolkit/nvcc missing. A PyTorch runtime wheel alone is insufficient")
    if not shutil.which("ninja"):
        raise RuntimeError("Install ninja: python -m pip install ninja")
    root = Path(__file__).resolve().parents[1]
    src = root / "assignments" / ASSIGNMENTS[assignment]["module"] / "csrc"
    if not src.exists():
        raise ValueError(f"{assignment} has no CUDA C++ extension")
    # Architecture-specific flags can be selected by TORCH_CUDA_ARCH_LIST.
    # Do not force an H100 binary onto Blackwell, or target all CUDA devices.
    return load(
        name=f"mgpu_{assignment}",
        sources=[str(src / "bindings.cpp"), str(src / "kernels.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v"],
        verbose=os.environ.get("MGPU_BUILD_VERBOSE", "0") == "1",
    )
