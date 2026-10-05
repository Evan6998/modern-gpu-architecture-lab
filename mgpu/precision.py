from contextlib import contextmanager
import torch


@contextmanager
def strict_reference_precision():
    """Restore process settings even if a candidate kernel fails."""
    backend = torch.backends.cuda.matmul
    old_tf32 = backend.allow_tf32
    old_reduced = backend.allow_fp16_reduced_precision_reduction
    backend.allow_tf32 = False
    backend.allow_fp16_reduced_precision_reduction = False
    try:
        yield
    finally:
        backend.allow_tf32 = old_tf32
        backend.allow_fp16_reduced_precision_reduction = old_reduced
