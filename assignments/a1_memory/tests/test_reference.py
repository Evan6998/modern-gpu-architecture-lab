import pytest
import torch
from .. import reference,student


@pytest.mark.parametrize("shape",[(1,1),(3,5),(31,65),(65,31)])
def test_transpose_oracle(shape):
    x=torch.arange(shape[0]*shape[1],dtype=torch.float32).reshape(shape)
    y=reference.transpose(x)
    assert y.is_contiguous() and y.shape==(shape[1],shape[0])
    for i in range(shape[0]):
        for j in range(shape[1]):assert y[j,i]==x[i,j]


@pytest.mark.parametrize("width",[1,31,32,33,129,1024,8193])
def test_softmax_uniform_and_shift(width):
    x=torch.zeros(2,width)
    y=reference.softmax(x)
    torch.testing.assert_close(y,torch.full_like(y,1/width))
    torch.testing.assert_close(reference.softmax(x+10000),y)
    torch.testing.assert_close(y.sum(-1),torch.ones(2))


def test_contract_rejects_wrong_output_shape():
    with pytest.raises(ValueError):student.transpose(torch.zeros(3,5),torch.zeros(3,5))


def test_contract_rejects_noncontiguous_input():
    with pytest.raises(ValueError):student.transpose(torch.zeros(3,5).t(),torch.zeros(3,5))


def test_contract_rejects_alias():
    x=torch.zeros(4,4)
    with pytest.raises(ValueError):student.softmax(x,x)


def test_contract_rejects_dtype():
    with pytest.raises(TypeError):student.softmax(torch.zeros(2,4).half(),torch.zeros(2,4).half())
