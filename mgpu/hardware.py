import torch


def compatible(family: str, cc: tuple[int, int]) -> bool:
    # Explicit allow-lists: future SM versions do not automatically inherit
    # architecture-specific instructions (especially sm_90a / sm_100a).
    if family == "cuda":
        return cc >= (7, 0)
    if family == "ampere":
        return cc[0] in (8, 9, 10, 12)
    if family == "hopper":
        return cc == (9, 0)
    if family == "blackwell":
        return cc == (10, 0)  # SM100 lab target, NOT SM120 RTX Blackwell.
    raise ValueError(f"Unknown architecture family: {family}")


def require(family: str, device=None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA is unavailable; run CPU reference checks instead")
    cc = torch.cuda.get_device_capability(device)
    if not compatible(family, cc):
        raise RuntimeError(f"{family} lab is not enabled for SM {cc[0]}.{cc[1]}; see docs/HARDWARE.md")
