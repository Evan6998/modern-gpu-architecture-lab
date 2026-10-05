"""Supplied correctness-oriented quantizers, NOT optimized production packers.

A and B^T are canonical K-major logical matrices [rows,K]. FP8 uses E4M3FN
with a per-tensor FP32 scale. MXFP4 uses E2M1 values and one E8M0 power-of-two
scale per 32 K-elements. This is MXFP4, NOT NVFP4's different scaling scheme.
"""
from dataclasses import dataclass
from typing import Any
import torch

FORMATS = ("fp8", "mxfp4")
LEVELS = (0., 0.5, 1., 1.5, 2., 3., 4., 6.)

@dataclass(frozen=True)
class QuantizedMatrix:
    data: torch.Tensor
    scales: torch.Tensor
    logical_shape: tuple[int, int]
    format: str

    def validate(self):
        if self.format not in FORMATS:
            raise ValueError(f"Unsupported quantization format: {self.format}")
        rows, k = self.logical_shape
        if rows <= 0 or k <= 0:
            raise ValueError("Logical dimensions must be positive")
        if self.data.device != self.scales.device or not self.data.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError("Packed data/scales must be contiguous on the same device")
        if self.format == "fp8":
            if self.data.dtype != torch.float8_e4m3fn or tuple(self.data.shape) != (rows, k):
                raise ValueError("FP8 payload must be E4M3FN [rows,K]")
            if self.scales.dtype != torch.float32 or tuple(self.scales.shape) != (1,):
                raise ValueError("FP8 scale must be FP32 [1]")
        else:
            if k % 32:
                raise ValueError("MXFP4 K must be divisible by 32")
            if self.data.dtype != torch.uint8 or tuple(self.data.shape) != (rows, k // 2):
                raise ValueError("MXFP4 payload must be uint8 [rows,K/2]")
            if self.scales.dtype != torch.uint8 or tuple(self.scales.shape) != (rows, k // 32):
                raise ValueError("MXFP4 E8M0 scales must be uint8 [rows,K/32]")
        return self

@dataclass
class PreparedQuantizedGemm:
    a: QuantizedMatrix
    bt: QuantizedMatrix
    backend: Any = None  # optional device-native scale/layout packing

    def validate(self):
        self.a.validate(); self.bt.validate()
        if self.a.format != self.bt.format or self.a.logical_shape[1] != self.bt.logical_shape[1]:
            raise ValueError("Quantized A and B^T have incompatible format/K")
        if self.a.data.device != self.bt.data.device:
            raise ValueError("Both operands must be on the same device")
        return self


def quantize_reference(x: torch.Tensor, format: str) -> QuantizedMatrix:
    if x.ndim != 2 or min(x.shape) <= 0 or x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Quantizer expects a non-empty FP16/FP32 matrix")
    if not torch.isfinite(x).all():
        raise ValueError("Quantizer requires finite input")
    f = x.float().contiguous()
    rows, k = f.shape
    if format == "fp8":
        max_abs = f.abs().max().reshape(1)
        scale = torch.where(max_abs > 0, max_abs / 448., torch.ones_like(max_abs))
        data = (f / scale).clamp(-448., 448.).to(torch.float8_e4m3fn)
        return QuantizedMatrix(data, scale, (rows, k), format).validate()
    if format != "mxfp4" or k % 32:
        raise ValueError("MXFP4 requires format='mxfp4' and K divisible by 32")
    blocks = f.reshape(rows, k // 32, 32)
    max_abs = blocks.abs().amax(dim=-1)
    exponent = torch.where(max_abs > 0, torch.ceil(torch.log2(max_abs.clamp_min(1e-38) / 6.)),
                           torch.zeros_like(max_abs)).clamp(-127, 127)
    scale = torch.pow(2., exponent)
    scaled = blocks / scale.unsqueeze(-1)
    levels = torch.tensor(LEVELS, dtype=torch.float32, device=f.device)
    distances = (scaled.abs().unsqueeze(-1) - levels).abs()
    codes = distances.argmin(-1)
    # Explicit round-to-nearest, ties-to-even in the E2M1 code space.
    next_code = (codes + 1).clamp_max(7)
    d0 = distances.gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    d1 = distances.gather(-1, next_code.unsqueeze(-1)).squeeze(-1)
    codes = torch.where((d0 == d1) & (codes % 2 == 1), next_code, codes)
    codes = (codes.to(torch.uint8) | ((scaled < 0).to(torch.uint8) << 3)).reshape(rows, k)
    data = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales = (exponent + 127).to(torch.uint8).contiguous()
    return QuantizedMatrix(data, scales, (rows, k), format).validate()


def dequantize_reference(q: QuantizedMatrix) -> torch.Tensor:
    q.validate()
    if q.format == "fp8":
        return q.data.float() * q.scales
    rows, k = q.logical_shape
    codes = torch.stack((q.data & 15, q.data >> 4), dim=-1).reshape(rows, k)
    levels = torch.tensor(LEVELS, dtype=torch.float32, device=q.data.device)
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)
    scales = torch.pow(2., q.scales.float() - 127.)
    return (values.reshape(rows, k // 32, 32) * scales.unsqueeze(-1)).reshape(rows, k)


def quantized_gemm_reference(prepared: PreparedQuantizedGemm) -> torch.Tensor:
    prepared.validate()
    a = dequantize_reference(prepared.a)
    bt = dequantize_reference(prepared.bt)
    return (a @ bt.t()).to(torch.float16)
