import pytest
import torch
from .. import reference,student


def test_gemm_known_answer():
    a=torch.tensor([[1,2],[3,4]],dtype=torch.float16)
    b=torch.tensor([[5,6],[7,8]],dtype=torch.float16)
    torch.testing.assert_close(reference.gemm(a,b),torch.tensor([[19,22],[43,50]],dtype=torch.float16))


def test_gemm_zero():
    out=reference.gemm(torch.zeros(3,7).half(),torch.randn(7,5).half())
    assert out.shape==(3,5) and torch.count_nonzero(out)==0


def test_shape_mismatch():
    with pytest.raises(ValueError):student.gemm(torch.zeros(3,7).half(),torch.zeros(6,5).half(),torch.zeros(3,5).half())


def test_output_dtype():
    with pytest.raises(TypeError):student.gemm(torch.zeros(128,32).half(),torch.zeros(32,128).half(),torch.zeros(128,128))


def test_no_autograd():
    a=torch.zeros(128,32,dtype=torch.float16,requires_grad=True)
    with pytest.raises(ValueError):student.gemm(a,torch.zeros(32,128).half(),torch.zeros(128,128).half())


def test_alignment_rejected():
    with pytest.raises(ValueError):student.gemm(torch.zeros(37,32).half(),torch.zeros(32,128).half(),torch.zeros(37,128).half())
