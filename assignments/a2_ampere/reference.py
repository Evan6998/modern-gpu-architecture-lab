import torch


def gemm(a, b):
    """FP32 accumulation of the already-rounded FP16 inputs."""
    return (a.float() @ b.float()).to(torch.float16)
