"""Independent, deliberately unoptimized oracle; never used by student code."""
import torch


def transpose(x):
    return x.t().contiguous()


def softmax(x):
    # Float64 oracle avoids sharing the student's reduction and exp path.
    return torch.softmax(x.double(), dim=-1).to(x.dtype)
