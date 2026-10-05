import pytest
import torch
from mgpu.hardware import compatible
from mgpu.precision import strict_reference_precision


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", help="Opt into student GPU tests")
    parser.addoption("--require-gpu", action="store_true", help="Fail if no CUDA GPU is present")
    parser.addoption("--strict-hardware", action="store_true", help="Fail instead of skip on wrong GPU architecture")


def pytest_sessionstart(session):
    if session.config.getoption("--require-gpu") and not torch.cuda.is_available():
        raise pytest.UsageError("CUDA unavailable: GPU grading cannot be replaced with CPU checks")


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu"):
        if not item.config.getoption("--run-gpu"):
            pytest.skip("GPU tests require --run-gpu (these skips are NOT assignment credit)")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        marker = item.get_closest_marker("arch")
        family = marker.args[0] if marker else "cuda"
        cc = torch.cuda.get_device_capability()
        if not compatible(family, cc):
            message = f"Requires {family}, detected SM {cc}; see docs/HARDWARE.md"
            if item.config.getoption("--strict-hardware"):
                pytest.fail(message)
            pytest.skip(message)


@pytest.fixture(scope="session", autouse=True)
def reproducible_reference_settings():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(old_threads, 4))
    with strict_reference_precision():
        yield
    torch.set_num_threads(old_threads)


@pytest.fixture
def cuda_device():
    return torch.device("cuda", torch.cuda.current_device())
