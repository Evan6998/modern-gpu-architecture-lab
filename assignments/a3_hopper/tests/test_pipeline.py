import pytest
import torch
from ..pipeline_model import ticket
from ..reference import cluster_copy


@pytest.mark.parametrize("stages",[1,2,3,8])
def test_ring_epochs(stages):
    seen=[ticket(i,stages) for i in range(stages*4)]
    for i,t in enumerate(seen):
        assert t.stage==i%stages and t.epoch==i//stages and t.parity==(i//stages)%2


def test_bad_ring_arguments():
    with pytest.raises(ValueError):ticket(-1,2)
    with pytest.raises(ValueError):ticket(0,0)


def test_cluster_oracle():
    x=torch.arange(128*128,dtype=torch.float32).reshape(128,128)
    out=cluster_copy(x)
    assert out.shape==(2,128,128)
    torch.testing.assert_close(out[0],x);torch.testing.assert_close(out[1],x)
