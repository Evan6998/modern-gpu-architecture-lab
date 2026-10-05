"""Local numerical oracle and deliberately inefficient distributed baseline."""
import torch
import torch.distributed as dist


def moe(x, all_weights, routes):
    if routes.dtype != torch.int64 or routes.shape != (x.shape[0],):
        raise ValueError("Expected int64 routes[T]")
    if routes.numel() and (routes.min() < 0 or routes.max() >= all_weights.shape[0]):
        raise ValueError("Expert IDs out of range")
    out = torch.empty((x.shape[0], all_weights.shape[2]), dtype=x.dtype, device=x.device)
    for expert in range(all_weights.shape[0]):
        idx = torch.nonzero(routes == expert, as_tuple=True)[0]
        if idx.numel():
            out[idx] = (x[idx].float() @ all_weights[expert].float()).to(x.dtype)
    return out


def distributed_into(x, local_weights, routes, out, *, group=None):
    world = dist.get_world_size(group)
    shards = [torch.empty_like(local_weights) for _ in range(world)]
    dist.all_gather(shards, local_weights, group=group)
    out.copy_(moe(x, torch.cat(shards, dim=0), routes))
    # This all-gather baseline is for harness validation ONLY. Student solutions
    # must dispatch tokens to expert owners, not replicate all expert weights.
