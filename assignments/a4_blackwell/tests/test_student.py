import pytest
import torch
from mgpu.bench import rand
from mgpu.checks import assert_output
from mgpu.test_support import exercise_gemm
from mgpu.quantization import quantize_reference,PreparedQuantizedGemm,quantized_gemm_reference
from .. import student
pytestmark=[pytest.mark.gpu,pytest.mark.student,pytest.mark.arch("blackwell")]


@pytest.mark.parametrize("variant",["tcgen05_1sm","tcgen05_2sm","persistent"])
@pytest.mark.parametrize("dims",[(128,128,64),(256,128,256),(128,384,160),(256,256,512)])
def test_dense(variant,dims,cuda_device):
    exercise_gemm("a4",dims,variant,cuda_device,repetitions=3)


@pytest.mark.parametrize("format",["fp8","mxfp4"])
@pytest.mark.parametrize("pattern",["random","zero","mixed_scales"])
# K=64 is a single MXFP4 MMA step on a single output tile; the larger shapes
# make the K loop, the tile grid and the per-tile scale layout observable.
@pytest.mark.parametrize("dims",[(128,128,64),(256,128,128),(128,384,192)])
def test_quantized(format,pattern,dims,cuda_device):
    m,n,k=dims
    a=rand((m,k),cuda_device,torch.float16,123)
    bt=rand((n,k),cuda_device,torch.float16,127)
    if pattern=="zero":a.zero_()
    if pattern=="mixed_scales":a[:,32:].mul_(8);bt[:,:32].mul_(.125)
    qa,qb=quantize_reference(a,format),quantize_reference(bt,format)
    original=[t.clone() for t in (qa.data,qa.scales,qb.data,qb.scales)]
    expected=quantized_gemm_reference(PreparedQuantizedGemm(qa,qb))
    prepared=student.prepare_quantized(qa,qb)
    out=torch.full_like(expected,float("nan"))
    student.quant_gemm(prepared,out)
    assert_output(out,expected,atol=.03,rtol=.01)
    # Compare bytes: torch arithmetic on float8 is intentionally limited.
    for t,old in zip((qa.data,qa.scales,qb.data,qb.scales),original):
        assert torch.equal(t.view(torch.uint8),old.view(torch.uint8))
