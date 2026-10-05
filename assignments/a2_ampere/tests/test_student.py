import pytest
from mgpu.test_support import exercise_gemm
pytestmark=[pytest.mark.gpu,pytest.mark.student,pytest.mark.arch("ampere")]


@pytest.mark.parametrize("variant",["simt","mma","async"])
@pytest.mark.parametrize("dims",[(1,1,1),(37,53,29),(128,128,32),(128,256,128),(256,128,256)])
def test_gemm(variant,dims,cuda_device):
    exercise_gemm("a2",dims,variant,cuda_device)


@pytest.mark.parametrize("stages",[1,2,3])
@pytest.mark.parametrize("k",[32,96,288])
def test_async_wraparound(stages,k,cuda_device):
    exercise_gemm("a2",(128,128,k),"async",cuda_device,{"stages":stages},repetitions=3)
