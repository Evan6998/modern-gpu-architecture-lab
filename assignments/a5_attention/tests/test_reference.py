import pytest
import torch
from .. import reference,student


@pytest.mark.parametrize("s",[1,3,17])
def test_zero_logits_causal_prefix_average(s):
    q=torch.zeros(1,1,s,128).half();k=q.clone()
    v=torch.arange(s).view(1,1,s,1).expand(-1,-1,-1,128).contiguous().half()
    out=reference.attention(q,k,v)
    expected=(torch.arange(s).float()/2).view(1,1,s,1).expand_as(out).half()
    torch.testing.assert_close(out,expected,atol=1e-3,rtol=1e-3)


def test_future_tokens_do_not_affect_past():
    g=torch.Generator().manual_seed(11)
    q,k,v=[torch.randn(1,2,17,128,generator=g).half() for _ in range(3)]
    first=reference.attention(q,k,v)
    k[:,:,8:]*=2;v[:,:,8:]+=10
    second=reference.attention(q,k,v)
    torch.testing.assert_close(first[:,:,:8],second[:,:,:8],atol=0,rtol=0)


def test_peaked_logits_select_one_past_key():
    # One-hot keys and a large one-hot query give an analytic answer: row i puts
    # all but ~1e-6 of its weight on key i//2, which is never in its future.
    s=64;target=torch.arange(s)//2
    k=torch.eye(s,128).view(1,1,s,128).half()
    q=(200*torch.eye(s,128)[target]).view(1,1,s,128).half()
    v=torch.randn(1,1,s,128,generator=torch.Generator().manual_seed(5)).half()
    torch.testing.assert_close(reference.attention(q,k,v),v[:,:,target],atol=1e-3,rtol=1e-3)


def test_invalid_head_dimension():
    x=torch.zeros(1,2,17,64).half()
    with pytest.raises(ValueError):student.attention(x,x.clone(),x.clone(),torch.empty_like(x))
