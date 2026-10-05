"""Edit csrc/kernels.cu; the Python API and output-buffer contract stay fixed."""
from mgpu import contracts
from mgpu.hardware import require
from mgpu.extension import load_extension


def transpose(x, out, *, variant="naive", config=None):
    contracts.transpose(x, out)
    names = ("naive", "tiled", "padded")
    if variant not in names:
        raise ValueError(f"Unknown transpose variant: {variant}")
    require("cuda", x.device)
    load_extension("a1").transpose(x, out, names.index(variant))


def softmax(x, out, *, variant="shuffle", config=None):
    contracts.softmax(x, out)
    names = ("shared", "shuffle")
    if variant not in names:
        raise ValueError(f"Unknown softmax variant: {variant}")
    require("cuda", x.device)
    load_extension("a1").softmax(x, out, names.index(variant))
