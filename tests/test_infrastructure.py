import csv
import json
from pathlib import Path
import sys
import types
import pytest
import torch
from mgpu import bench,grade
from mgpu.config import KernelConfig
from mgpu.hardware import compatible,require
from mgpu.timing import Timing,measure
from mgpu.checks import assert_output,error_metrics,TOLERANCES
from mgpu.registry import ASSIGNMENTS,module
from mgpu.bench import append_csv,COLUMNS,prepare_case
from mgpu.grade import junit_counts


def test_pybind11_headers_found_when_torch_does_not_bundle_them(monkeypatch,tmp_path):
    from mgpu.extension import pybind11_include_paths
    bundled=tmp_path/"wheel";(bundled/"include"/"pybind11").mkdir(parents=True)
    assert pybind11_include_paths(bundled)==[]
    distro=tmp_path/"distro";(distro/"include").mkdir(parents=True)
    monkeypatch.setitem(sys.modules,"pybind11",types.SimpleNamespace(get_include=lambda:"/somewhere/pybind11/include"))
    assert pybind11_include_paths(distro)==["/somewhere/pybind11/include"]
    monkeypatch.setitem(sys.modules,"pybind11",None)  # import now raises ImportError
    assert pybind11_include_paths(distro)==[]


def test_registry_imports_without_cuda_compilation():
    for name in ASSIGNMENTS:
        module(name,"student");module(name,"reference")


@pytest.mark.parametrize("cc,family,expected",[
    ((8,0),"ampere",True),((9,0),"hopper",True),((10,0),"blackwell",True),
    ((12,0),"blackwell",False),((10,0),"hopper",False),((9,0),"blackwell",False),
    ((13,0),"ampere",False),((7,5),"ampere",False),((7,5),"cuda",True)])
def test_architecture_gates(cc,family,expected):
    assert compatible(family,cc)==expected


def test_cpu_tensors_rejected_the_same_way_on_a_gpu_machine(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    with pytest.raises(RuntimeError,match="CUDA tensors"):require("cuda",torch.device("cpu"))


@pytest.mark.parametrize("bad",[{"stages":0},{"stages":9},{"warps":-1},{"tile_m":1.5},{"stages":True}])
def test_bad_config(bad):
    with pytest.raises(ValueError):KernelConfig.parse(bad)


def test_unknown_config_rejected():
    with pytest.raises(TypeError):KernelConfig.parse({"typo":4})


def test_timing_statistics():
    t=Timing([3.,1.,2.],"test","warm")
    assert t.p50_ms==2.
    assert t.percentile(0)==1.
    assert t.percentile(1)==3.
    assert t.percentile(.5)==2.


def test_cpu_timer_calls_and_labels():
    calls=[]
    t=measure(lambda:calls.append(1),device=torch.device("cpu"),warmup=2,repeats=3)
    assert len(calls)==5 and len(t.samples_ms)==3
    assert t.scope=="cpu_reference_wall_time"


@pytest.mark.parametrize("kwargs",[{"repeats":0},{"warmup":0},{"graph":True},{"evict_mb":1}])
def test_bad_cpu_timer_arguments(kwargs):
    with pytest.raises(ValueError):measure(lambda:None,device=torch.device("cpu"),**kwargs)


def test_cuda_workspace_accounting_excludes_eviction_buffer(monkeypatch):
    """Drive measure()'s CUDA branch with a fake allocator; no GPU involved."""
    import mgpu.timing as timing
    mib=1024*1024;state={"allocated":0,"peak":0}
    def alloc(n):
        state["allocated"]+=n;state["peak"]=max(state["peak"],state["allocated"])
    class Buffer:
        def fill_(self,value):return self
    class Event:
        def __init__(self,enable_timing=False):pass
        def record(self):pass
        def synchronize(self):pass
        def elapsed_time(self,other):return 1.
    class Device:
        def __init__(self,device):pass
        def __enter__(self):return self
        def __exit__(self,*exc):return False
    def empty(n,**kwargs):
        alloc(n);return Buffer()
    cuda=types.SimpleNamespace(synchronize=lambda device=None:None,device=Device,Event=Event,
        memory_allocated=lambda device=None:state["allocated"],
        max_memory_allocated=lambda device=None:state["peak"],
        reset_peak_memory_stats=lambda device=None:state.__setitem__("peak",state["allocated"]))
    monkeypatch.setattr(timing,"torch",types.SimpleNamespace(cuda=cuda,empty=empty,uint8=torch.uint8))
    def fn():  # an operator with a transient 3 MiB workspace
        alloc(3*mib);state["allocated"]-=3*mib
    t=timing.measure(fn,device=torch.device("cuda"),warmup=1,repeats=2,evict_mb=64)
    assert t.peak_extra_bytes==3*mib
    assert t.scope=="cuda_events_operator" and t.cache_policy=="eviction_buffer_64MiB"
    assert len(t.samples_ms)==2


def test_cpu_timer_reports_no_gpu_workspace():
    assert measure(lambda:None,device=torch.device("cpu"),warmup=1,repeats=1).peak_extra_bytes is None


def test_nan_output_fails():
    with pytest.raises(AssertionError):assert_output(torch.tensor([float("nan")]),torch.ones(1))


def test_wrong_dtype_fails():
    with pytest.raises(AssertionError):assert_output(torch.ones(1).half(),torch.ones(1))


def test_error_metrics():
    e=error_metrics(torch.tensor([2.,2.]),torch.tensor([1.,1.]))
    assert e=={"max_abs":1.,"relative_rmse":1.}


def test_csv_append_and_schema_guard(tmp_path):
    path=tmp_path/"bench.csv";row={k:None for k in COLUMNS};row["run_id"]="test"
    append_csv(path,row);append_csv(path,row)
    with path.open() as f:assert len(list(csv.DictReader(f)))==2
    path.write_text("wrong,header\n")
    with pytest.raises(ValueError):append_csv(path,row)


def test_junit_counts(tmp_path):
    path=tmp_path/"junit.xml"
    path.write_text('<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1"/></testsuites>')
    assert junit_counts(path)=={"tests":3,"failures":1,"errors":0,"skipped":1}


@pytest.mark.parametrize("aid,op,variant,dims",[
    ("a1","transpose","naive",[3,5]),("a1","softmax","shared",[2,33]),
    ("a2","gemm","simt",[8,8,16]),("a3","gemm","tma_wgmma",[128,128,32]),
    ("a3","cluster_copy","dsm",[128,128]),("a4","gemm","tcgen05_1sm",[128,128,32]),
    ("a4","quant_gemm","fp8",[128,128,32]),("a4","quant_gemm","mxfp4",[128,128,32]),
    ("a5","attention","fused",[1,1,17,128])])
def test_cpu_benchmark_preparation(aid,op,variant,dims):
    fn,out,expected,flops,io_bytes,err=prepare_case(aid,op,dims,"reference",variant,torch.device("cpu"),{},"prepacked")
    fn();assert_output(out,expected)
    assert flops>=0 and io_bytes>0


def test_every_benchmarked_operation_has_a_tolerance():
    for name,spec in ASSIGNMENTS.items():
        if name!="a6":assert set(spec["ops"])<=set(TOLERANCES),name


@pytest.mark.parametrize("argv,label",[
    (["a1","--op","transpose"],"n/a"),(["a3","--op","gemm"],"n/a"),
    (["a4","--op","quant_gemm","--variant","mxfp4"],"mxfp4")])
def test_bench_cli_labels_reference_rows(tmp_path,argv,label):
    bench.main(argv+["--impl","reference","--device","cpu","--suite","smoke",
                     "--warmup","1","--repeats","2","--output",str(tmp_path)])
    with (tmp_path/"bench.csv").open() as f:rows=list(csv.DictReader(f))
    assert [r["variant"] for r in rows]==[label]
    assert rows[0]["scope"]=="cpu_reference_wall_time" and rows[0]["peak_extra_allocated_bytes"]==""


def test_bench_attention_uses_peaked_logits():
    dims=[1,1,129,128]
    fn,out,expected,*_=prepare_case("a5","attention",dims,"reference","fused",torch.device("cpu"),{},"prepacked")
    q,k,v=[bench.rand(dims,"cpu",torch.float16,seed) for seed in (17,23,29)]
    q=q*bench.ATTENTION_Q_GAIN
    logits=q.float()@k.float().transpose(-1,-2)/128**.5
    assert 1<float(logits.std())<4  # far from uniform; rand() alone gives ~0.06
    torch.testing.assert_close(expected,module("a5","reference").attention(q,k,v),atol=0,rtol=0)


def test_grade_rejects_unknown_variant(capsys):
    with pytest.raises(SystemExit):grade.main(["a1","--variant","nieve"])
    assert "naive" in capsys.readouterr().err


def test_manifest_complete():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"course.json").read_text())
    assert {a["id"] for a in manifest["assignments"]}==set(ASSIGNMENTS)
    for a in manifest["assignments"]:
        assert sum(a["rubric"].values())==100
        path=root/a["directory"]
        for f in ("README.md","student.py","reference.py","benchmark.json","REPORT.md","submission.json"):
            assert (path/f).is_file(),path/f
