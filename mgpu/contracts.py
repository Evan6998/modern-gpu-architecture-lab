"""Host-side API contracts. Validate BEFORE allocating or launching kernels."""
import torch


def tensors(*xs: torch.Tensor, dtype: torch.dtype | None = None) -> None:
    if not xs or any(not isinstance(x, torch.Tensor) for x in xs):
        raise TypeError("Expected torch.Tensor inputs")
    if len({x.device for x in xs}) != 1:
        raise ValueError("All tensors must be on the same device")
    for x in xs:
        if not x.is_contiguous():
            raise ValueError("Only contiguous row-major tensors are supported")
        if x.requires_grad:
            raise ValueError("This forward-only lab does not support autograd")
        if dtype is not None and x.dtype != dtype:
            raise TypeError(f"Expected {dtype}; got {x.dtype}")


def distinct_output(out: torch.Tensor, *inputs: torch.Tensor) -> None:
    if out.numel() and any(x.numel() and out.untyped_storage().data_ptr() ==
                           x.untyped_storage().data_ptr() for x in inputs):
        raise ValueError("Output may not alias any input storage")


def shape(t: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(t.shape) != expected:
        raise ValueError(f"Expected shape {expected}; got {tuple(t.shape)}")


def matrix(x: torch.Tensor) -> None:
    if x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError("Expected a non-empty 2-D matrix")


def transpose(x, out):
    tensors(x, out, dtype=torch.float32)
    matrix(x)
    shape(out, (x.shape[1], x.shape[0]))
    distinct_output(out, x)


def softmax(x, out):
    tensors(x, out, dtype=torch.float32)
    matrix(x)
    shape(out, tuple(x.shape))
    distinct_output(out, x)


def gemm(a, b, out, *, aligned=False):
    tensors(a, b, out, dtype=torch.float16)
    matrix(a); matrix(b)
    if a.shape[1] != b.shape[0]:
        raise ValueError("A[M,K] and B[K,N] have incompatible K")
    shape(out, (a.shape[0], b.shape[1]))
    distinct_output(out, a, b)
    if aligned and (a.shape[0] % 128 or b.shape[1] % 128 or a.shape[1] % 32):
        raise ValueError("A3/A4 core contract: M,N multiples of 128; K multiple of 32")


def cluster_copy(x, out):
    tensors(x, out, dtype=torch.float32)
    shape(x, (128, 128)); shape(out, (2, 128, 128))
    distinct_output(out, x)


def attention(q, k, v, out):
    tensors(q, k, v, out, dtype=torch.float16)
    if q.ndim != 4 or q.shape[-1] != 128 or min(q.shape) <= 0:
        raise ValueError("Expected non-empty [B,H,S,128]")
    for t in (k, v, out):
        shape(t, tuple(q.shape))
    distinct_output(out, q, k, v)


def moe(x, local_weights, routes, out, *, world_size):
    tensors(x, local_weights, out, dtype=torch.float16)
    tensors(x, routes)
    if world_size not in (1, 2, 4, 8):
        raise ValueError("Eight experts require world_size in {1,2,4,8}")
    if x.ndim != 2 or local_weights.ndim != 3:
        raise ValueError("Expected X[T,D], local_weights[8/world_size,D,H]")
    if local_weights.shape[0] != 8 // world_size or x.shape[1] != local_weights.shape[1]:
        raise ValueError("Invalid local expert shard")
    if x.shape[1] <= 0 or local_weights.shape[2] <= 0:
        raise ValueError("D,H must be positive")
    if routes.dtype != torch.int64:
        raise TypeError("routes must be int64")
    shape(routes, (x.shape[0],)); shape(out, (x.shape[0], local_weights.shape[2]))
    distinct_output(out, x, local_weights, routes)
    # Range validation belongs to dataset construction, outside the timed path:
    # checking routes.min().item() on CUDA would insert a host synchronization.
