import pytest
import torch
from ..reference import moe
from ..run import make_data


@pytest.mark.parametrize("tokens",[0,1,17,33])
def test_expert_identity_and_order(tokens):
    x=torch.arange(tokens*4).reshape(tokens,4).half()/16
    weights=torch.stack([torch.eye(4)*(e+1) for e in range(8)]).half()
    routes=torch.arange(tokens).flip(0)%8
    expected=x*(routes[:,None]+1).half()
    torch.testing.assert_close(moe(x,weights,routes),expected,atol=0,rtol=0)


@pytest.mark.parametrize("world",[1,2,4,8])
@pytest.mark.parametrize("case",["uniform","skew","all_to_one","uneven","empty"])
def test_public_distributed_dataset(case,world):
    shards=[]
    for rank in range(world):
        x,local,routes,all_weights=make_data(case,rank,world,17,8,4,torch.device("cpu"))
        assert local.shape==(8//world,8,4)
        if case=="uneven" and rank==0:assert x.shape[0]==0
        if case=="empty":assert x.shape[0]==0
        assert moe(x,all_weights,routes).shape==(x.shape[0],4)
        shards.append(local)
    torch.testing.assert_close(torch.cat(shards),all_weights,atol=0,rtol=0)


def test_invalid_route():
    with pytest.raises(ValueError):moe(torch.ones(1,4).half(),torch.ones(8,4,3).half(),torch.tensor([8]))


def test_shape_and_shard_contract():
    from mgpu.contracts import moe as validate
    with pytest.raises(ValueError):validate(torch.ones(1,4).half(),torch.ones(3,4,3).half(),torch.tensor([0]),torch.zeros(1,3).half(),world_size=2)
