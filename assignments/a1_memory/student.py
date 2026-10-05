"""Edit csrc/kernels.cu; the Python API and output-buffer contract stay fixed."""
from mgpu import contracts
from mgpu.config import KernelConfig
from mgpu.hardware import require
from mgpu.extension import load_extension

# The only KernelConfig fields A1 forwards to the kernels, with A1's own
# defaults. Any other key is rejected: a benchmark row must never record a
# knob that the kernel did not see.
TRANSPOSE_DEFAULTS = {"tile_m": 32}  # tile edge (naive: 2-D block edge)
SOFTMAX_DEFAULTS = {"warps": 4}      # warps per block


def _config(config, defaults):
    config = dict(config or {})
    unknown = sorted(set(config) - set(defaults))
    if unknown:
        raise ValueError(f"This A1 operation reads only {sorted(defaults)}; got {unknown}")
    return KernelConfig.parse({**defaults, **config})


def transpose(x, out, *, variant="naive", config=None):
    contracts.transpose(x, out)
    names = ("naive", "tiled", "padded")
    if variant not in names:
        raise ValueError(f"Unknown transpose variant: {variant}")
    cfg = _config(config, TRANSPOSE_DEFAULTS)
    require("cuda", x.device)
    load_extension("a1").transpose(x, out, names.index(variant), cfg.tile_m)


def softmax(x, out, *, variant="shuffle", config=None):
    contracts.softmax(x, out)
    names = ("shared", "shuffle")
    if variant not in names:
        raise ValueError(f"Unknown softmax variant: {variant}")
    cfg = _config(config, SOFTMAX_DEFAULTS)
    require("cuda", x.device)
    load_extension("a1").softmax(x, out, names.index(variant), cfg.warps)
