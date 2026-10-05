import torch.distributed as dist
from mgpu import contracts


def moe(x, local_weights, routes, out, *, variant="serial", config=None, group=None):
    if not dist.is_initialized():
        raise RuntimeError("Launch under torchrun; initialize the process group first")
    if variant not in ("serial", "overlap"):
        raise ValueError("Unknown MoE variant")
    if dist.get_backend(group) != "nccl" or x.device.type != "cuda":
        raise RuntimeError("Student MoE requires CUDA + NCCL; Gloo is reference-harness only")
    world = dist.get_world_size(group)
    contracts.moe(x, local_weights, routes, out, world_size=world)
    from mgpu.config import KernelConfig
    cfg = KernelConfig.parse(config)
    # Ownership: rank r owns [r*(8/world), (r+1)*(8/world)).
    # TODO(A6.1): count/pack tokens by destination; exchange variable split sizes.
    # TODO(A6.2): dispatch token data + expert/local-index metadata via A2A.
    # TODO(A6.3): execute only the locally owned expert GEMMs (reuse A2-A4).
    # TODO(A6.4): return results and restore original token order, including zero
    #            sends, empty local experts and ranks with no input tokens.
    # TODO(A6.5): overlap using chunks, streams and explicit event dependencies.
    # Annotate phases with NVTX; ensure internal streams rejoin CURRENT stream.
    # No gathering all expert weights; no dropped/padded-away real tokens.
    raise NotImplementedError(f"A6: implement {variant} top-1 dispatch/compute/return")
