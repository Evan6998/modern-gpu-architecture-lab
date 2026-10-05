from dataclasses import replace
import pytest
import torch
from mgpu.quantization import (quantize_reference,dequantize_reference,
    PreparedQuantizedGemm,quantized_gemm_reference)
from mgpu.checks import error_metrics
from .. import student


@pytest.mark.parametrize("format",["fp8","mxfp4"])
def test_zero_quantization(format):
    x=torch.zeros(4,64).half();q=quantize_reference(x,format)
    torch.testing.assert_close(dequantize_reference(q),x.float(),atol=0,rtol=0)
    assert q.data.device==x.device


def test_mxfp4_known_codebook_and_packing():
    values=[0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6]*2
    x=torch.tensor([values],dtype=torch.float32)
    q=quantize_reference(x,"mxfp4")
    assert torch.equal(q.scales,torch.tensor([[127]],dtype=torch.uint8))
    expected_codes=[0,1,2,3,4,5,6,7,0,9,10,11,12,13,14,15]*2
    expected_bytes=[a|(b<<4) for a,b in zip(expected_codes[::2],expected_codes[1::2])]
    assert q.data.tolist()==[expected_bytes]
    torch.testing.assert_close(dequantize_reference(q),x,atol=0,rtol=0)


def test_mxfp4_ties_to_even():
    x=torch.zeros(1,32);x[0,:8]=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.,6.])
    d=dequantize_reference(quantize_reference(x,"mxfp4"))
    torch.testing.assert_close(d[0,:8],torch.tensor([0.,1.,1.,2.,2.,4.,4.,6.]),atol=0,rtol=0)


def test_mxfp4_per_block_scale():
    x=torch.cat((torch.full((1,32),6.),torch.full((1,32),12.)),dim=1)
    q=quantize_reference(x,"mxfp4")
    assert q.scales.tolist()==[[127,128]]
    torch.testing.assert_close(dequantize_reference(q),x,atol=0,rtol=0)


@pytest.mark.parametrize("format,bound",[("fp8",.08),("mxfp4",.30)])
def test_quantization_error_separate_from_kernel_error(format,bound):
    g=torch.Generator().manual_seed(123)
    a=torch.randn(16,64,generator=g).half();bt=torch.randn(24,64,generator=g).half()
    p=PreparedQuantizedGemm(quantize_reference(a,format),quantize_reference(bt,format))
    qref=quantized_gemm_reference(p)
    original=(a.float()@bt.float().t()).half()
    assert qref.shape==(16,24)
    assert error_metrics(qref,original)["relative_rmse"]<bound


def test_invalid_packed_layout_rejected():
    q=quantize_reference(torch.ones(2,32),"mxfp4")
    with pytest.raises(ValueError):replace(q,data=q.data[:,:-1]).validate()


def test_nonfinite_input_rejected():
    with pytest.raises(ValueError):quantize_reference(torch.full((2,32),float("nan")),"fp8")


def test_invalid_k_rejected():
    with pytest.raises(ValueError):quantize_reference(torch.ones(2,33),"mxfp4")


def test_quant_gemm_k_granularity_follows_the_mma_instruction():
    out=torch.zeros(128,128).half()
    q4=quantize_reference(torch.ones(128,96),"mxfp4")
    with pytest.raises(ValueError,match="multiple of 64"):
        student.quant_gemm(PreparedQuantizedGemm(q4,q4),out)
    # FP8 keeps K%32: the same shape passes the contract and stops at the GPU gate.
    q8=quantize_reference(torch.ones(128,96),"fp8")
    with pytest.raises(RuntimeError):student.quant_gemm(PreparedQuantizedGemm(q8,q8),out)
