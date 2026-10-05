"""Single source of truth for names and public variants."""
from importlib import import_module

ASSIGNMENTS = {
    "a1": {"module": "a1_memory", "arch": "cuda",
           "ops": {"transpose": ["naive", "tiled", "padded"],
                   "softmax": ["shared", "shuffle"]}},
    "a2": {"module": "a2_ampere", "arch": "ampere",
           "ops": {"gemm": ["simt", "mma", "async"]}},
    "a3": {"module": "a3_hopper", "arch": "hopper",
           "ops": {"gemm": ["tma_wgmma"],
                   "cluster_copy": ["duplicate", "multicast", "dsm"]}},
    "a4": {"module": "a4_blackwell", "arch": "blackwell",
           "ops": {"gemm": ["tcgen05_1sm", "tcgen05_2sm", "persistent"],
                   "quant_gemm": ["fp8", "mxfp4"]}},
    "a5": {"module": "a5_attention", "arch": "ampere",
           "ops": {"attention": ["fused"]}},
    "a6": {"module": "a6_moe", "arch": "cuda",
           "ops": {"moe": ["serial", "overlap"]}},
}

def module(assignment: str, part: str):
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"Unknown assignment: {assignment}")
    return import_module(f"assignments.{ASSIGNMENTS[assignment]['module']}.{part}")
