import pytest
import torch
from mgpu.bench import rand
from mgpu.checks import assert_output
from .. import reference,student
pytestmark=[pytest.mark.gpu,pytest.mark.student,pytest.mark.arch("ampere")]


@pytest.mark.parametrize("s",[1,17,129,256,1024])
@pytest.mark.parametrize("pattern",["random","zero_logits"])
def test_fused_attention(s,pattern,cuda_device):
    q,k,v=[rand((1,2,s,128),cuda_device,torch.float16,seed) for seed in (17,19,23)]
    if pattern=="zero_logits":q.zero_()
    copies=[x.clone() for x in (q,k,v)]
    expected=reference.attention(q,k,v);out=torch.full_like(expected,float("nan"))
    student.attention(q,k,v,out)
    assert_output(out,expected,atol=.003,rtol=.01)
    for x,old in zip((q,k,v),copies):torch.testing.assert_close(x,old,atol=0,rtol=0)


def test_causal_invariance(cuda_device):
    q,k,v=[rand((1,2,129,128),cuda_device,torch.float16,seed) for seed in (17,19,23)]
    first=torch.empty_like(q);second=torch.empty_like(q)
    student.attention(q,k,v,first)
    k[:,:,64:].mul_(2);v[:,:,64:].add_(10)
    student.attention(q,k,v,second)
    torch.testing.assert_close(first[:,:,:64],second[:,:,:64],atol=.003,rtol=.01)


def test_no_quadratic_workspace(cuda_device):
    q,k,v=[rand((1,2,2048,128),cuda_device,torch.float16,seed) for seed in (17,19,23)]
    out=torch.empty_like(q)
    student.attention(q,k,v,out)  # JIT warmup outside allocation observation
    torch.cuda.synchronize(cuda_device)
    before=torch.cuda.memory_allocated(cuda_device)
    torch.cuda.reset_peak_memory_stats(cuda_device)
    student.attention(q,k,v,out);torch.cuda.synchronize(cuda_device)
    extra=torch.cuda.max_memory_allocated(cuda_device)-before
    one_score_matrix=1*2*2048*2048*4
    assert extra<one_score_matrix//2,"Workspace indicates an S*S intermediate; inspect with Nsight"
