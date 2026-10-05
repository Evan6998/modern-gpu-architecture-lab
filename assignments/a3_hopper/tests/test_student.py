import pytest
import torch
from mgpu.bench import rand
from mgpu.checks import assert_output
from mgpu.test_support import exercise_gemm
from .. import reference,student
pytestmark=[pytest.mark.gpu,pytest.mark.student,pytest.mark.arch("hopper")]


@pytest.mark.parametrize("stages",[1,2,3])
@pytest.mark.parametrize("k",[32,96,288])
def test_tma_wgmma(stages,k,cuda_device):
    exercise_gemm("a3",(128,128,k),"tma_wgmma",cuda_device,{"stages":stages},repetitions=4)


@pytest.mark.parametrize("variant",["duplicate","multicast","dsm"])
def test_cluster_copy(variant,cuda_device):
    for seed in range(3):
        x=rand((128,128),cuda_device,torch.float32,seed)
        out=torch.full((2,128,128),float("nan"),device=cuda_device)
        student.cluster_copy(x,out,variant=variant)
        assert_output(out,reference.cluster_copy(x),atol=0,rtol=0)
