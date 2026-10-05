import torch


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
