import csv
import json
from pathlib import Path
import pytest
import torch
from mgpu.config import KernelConfig
from mgpu.hardware import compatible
from mgpu.timing import Timing,measure
from mgpu.checks import assert_output,error_metrics
from mgpu.registry import ASSIGNMENTS,module
from mgpu.bench import append_csv,COLUMNS,prepare_case
from mgpu.grade import junit_counts


def test_registry_imports_without_cuda_compilation():
    for name in ASSIGNMENTS:
        module(name,"student");module(name,"reference")


@pytest.mark.parametrize("cc,family,expected",[
    ((8,0),"ampere",True),((9,0),"hopper",True),((10,0),"blackwell",True),
    ((12,0),"blackwell",False),((10,0),"hopper",False),((9,0),"blackwell",False),
    ((13,0),"ampere",False),((7,5),"ampere",False),((7,5),"cuda",True)])
def test_architecture_gates(cc,family,expected):
    assert compatible(family,cc)==expected


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


def test_manifest_complete():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"course.json").read_text())
    assert {a["id"] for a in manifest["assignments"]}==set(ASSIGNMENTS)
    for a in manifest["assignments"]:
        assert sum(a["rubric"].values())==100
        path=root/a["directory"]
        for f in ("README.md","student.py","reference.py","benchmark.json","REPORT.md","submission.json"):
            assert (path/f).is_file(),path/f
