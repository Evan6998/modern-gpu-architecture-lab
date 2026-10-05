import torch

# (atol, rtol) per operation: the bounds the public GPU tests use. The benchmark
# gate reads the same table, so it never times a result that grading would reject.
TOLERANCES = {
    "transpose": (0.0, 0.0),
    "cluster_copy": (0.0, 0.0),
    "softmax": (2e-6, 2e-5),
    "attention": (3e-3, 1e-2),
    "gemm": (0.03, 0.01),
    "quant_gemm": (0.03, 0.01),
}


def assert_output(actual, expected, *, atol=0.03, rtol=0.01):
    if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
        raise AssertionError("Output shape/type mismatch")
    if actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError("Output dtype/device mismatch")
    if not actual.is_contiguous():
        raise AssertionError("Output must be contiguous")
    if not torch.isfinite(actual).all():
        raise AssertionError("Output contains NaN or Inf (unwritten output is also a failure)")
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def error_metrics(actual, expected):
    a, b = actual.float(), expected.float()
    diff = a - b
    denom = b.square().mean().sqrt().clamp_min(1e-12)
    return {"max_abs": float(diff.abs().max()),
            "relative_rmse": float(diff.square().mean().sqrt() / denom)}
