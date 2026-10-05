import math
import torch


def attention(q, k, v):
    """Materialized FP32 causal attention oracle (quadratic workspace)."""
    s, d = q.shape[-2:]
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d)
    mask = torch.ones((s, s), dtype=torch.bool, device=q.device).triu(1)
    scores.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(q.dtype)


def materialized_into(q, k, v, out):
    out.copy_(attention(q, k, v))
