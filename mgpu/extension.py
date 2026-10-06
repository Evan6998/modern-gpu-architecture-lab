"""Lazy C++/CUDA build: importing the lab never invokes nvcc."""
from functools import lru_cache
from pathlib import Path
import os
import shutil


def pybind11_include_paths(torch_dir):
    """Extra include paths for PyTorch builds that do not bundle pybind11.

    pip wheels ship the headers under torch/include/pybind11. Distribution
    builds (for example an apt-packaged PyTorch) do not; there the separately
    installed ``pybind11`` module provides them.
    """
    if (Path(torch_dir) / "include" / "pybind11").exists():
        return []
    try:
        import pybind11
    except ImportError:
        return []  # system-wide headers may still exist; let the compiler decide
    return [pybind11.get_include()]


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
    # No -std flag here: torch.utils.cpp_extension supplies the standard its own
    # headers need (C++17 for older releases, C++20 for newer ones). Forcing
    # -std=c++17 overrides that and breaks the build against C++20-only headers.
    return load(
        name=f"mgpu_{assignment}",
        sources=[str(src / "bindings.cpp"), str(src / "kernels.cu")],
        extra_include_paths=pybind11_include_paths(Path(torch.__file__).parent),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
        verbose=os.environ.get("MGPU_BUILD_VERBOSE", "0") == "1",
    )
