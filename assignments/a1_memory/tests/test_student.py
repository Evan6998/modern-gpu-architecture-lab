import pytest
import torch
from mgpu.bench import rand
from mgpu.checks import assert_output
from .. import reference,student
pytestmark=[pytest.mark.gpu,pytest.mark.student,pytest.mark.arch("cuda")]


@pytest.mark.parametrize("variant",["naive","tiled","padded"])
@pytest.mark.parametrize("shape",[(1,1),(3,5),(31,65),(128,128),(257,513)])
def test_transpose(variant,shape,cuda_device):
    stream=torch.cuda.Stream(device=cuda_device)
    with torch.cuda.stream(stream):
        x=rand(shape,cuda_device,torch.float32,41);x0=x.clone()
        out=torch.full((shape[1],shape[0]),float("nan"),device=cuda_device)
        student.transpose(x,out,variant=variant);observed=out.clone()
    stream.synchronize()
    assert_output(observed,reference.transpose(x),atol=0,rtol=0)
    torch.testing.assert_close(x,x0,atol=0,rtol=0)


@pytest.mark.parametrize("variant",["shared","shuffle"])
@pytest.mark.parametrize("width",[1,31,32,33,127,128,129,1024,8192,8193])
def test_softmax(variant,width,cuda_device):
    x=rand((5,width),cuda_device,torch.float32,42)*30
    x[0].fill_(10000);x[1].fill_(-10000)
    x0=x.clone();out=torch.full_like(x,float("nan"))
    student.softmax(x,out,variant=variant)
    assert_output(out,reference.softmax(x),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(out.sum(-1),torch.ones(5,device=cuda_device),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(x,x0,atol=0,rtol=0)
