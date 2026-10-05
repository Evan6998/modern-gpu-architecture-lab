"""Shared benchmark CLI. Correctness is checked before any timing is recorded."""
import argparse
import csv
import json
import math
from pathlib import Path
import uuid
import torch
import torch.nn.functional as F
from mgpu.registry import ASSIGNMENTS, module
from mgpu.config import KernelConfig
from mgpu.hardware import require
from mgpu.doctor import snapshot
from mgpu.checks import assert_output, error_metrics
from mgpu.precision import strict_reference_precision
from mgpu.timing import measure
from mgpu.quantization import (quantize_reference, PreparedQuantizedGemm,
                              quantized_gemm_reference)

COLUMNS = ["run_id", "assignment", "op", "impl", "variant", "shape", "config",
           "device", "gpu_name", "sm", "scope", "cache_policy", "quant_scope",
           "warmup", "repeats", "p50_ms", "p10_ms", "p90_ms", "tflops_estimate",
           "logical_io_gbs_estimate", "peak_extra_allocated_bytes", "kernel_max_abs",
           "kernel_relative_rmse", "quantization_max_abs", "quantization_relative_rmse",
           "samples_ms"]


def rand(shape, device, dtype, seed):
    # CPU-side generator makes the public dataset identical across GPU families.
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float32) * 0.25).to(device=device, dtype=dtype)


def prepare_case(aid, op, dims, impl, variant, device, config, quant_scope):
    ref, student = module(aid, "reference"), module(aid, "student")
    quant_error = {"max_abs": None, "relative_rmse": None}
    if op == "transpose" or op == "softmax":
        x = rand(dims, device, torch.float32, 17)
        expected = getattr(ref, op)(x)
        out = torch.full_like(expected, float("nan"))
        if impl == "student":
            fn = lambda: getattr(student, op)(x, out, variant=variant, config=config)
        elif impl == "library":
            fn = (lambda: out.copy_(x.t())) if op == "transpose" else (lambda: torch.softmax(x, -1, out=out))
        else:
            fn = lambda: out.copy_(getattr(ref, op)(x))
        return fn, out, expected, 0, 2*x.numel()*4, quant_error
    if op == "gemm":
        m, n, k = dims
        a = rand((m,k),device,torch.float16,17); b = rand((k,n),device,torch.float16,23)
        expected = ref.gemm(a,b); out = torch.full_like(expected,float("nan"))
        if impl == "student":
            fn = lambda: student.gemm(a,b,out,variant=variant,config=config)
        elif impl == "library":
            fn = lambda: torch.mm(a,b,out=out)
        else:
            fn = lambda: out.copy_(ref.gemm(a,b))
        return fn,out,expected,2*m*n*k,2*(m*k+k*n+m*n),quant_error
    if op == "cluster_copy":
        x=rand(dims,device,torch.float32,17)
        expected=ref.cluster_copy(x);out=torch.full_like(expected,float("nan"))
        fn=(lambda: student.cluster_copy(x,out,variant=variant,config=config)) if impl=="student" else (lambda: out.copy_(x.unsqueeze(0).expand_as(out)))
        return fn,out,expected,0,(x.numel()+out.numel())*4,quant_error
    if op == "attention":
        q,k,v=[rand(dims,device,torch.float16,s) for s in (17,23,29)]
        expected=ref.attention(q,k,v);out=torch.full_like(expected,float("nan"))
        if impl=="student":
            fn=lambda: student.attention(q,k,v,out,variant=variant,config=config)
        elif impl=="library":
            # Allocation + copy are included; this is an operator baseline,
            # not a claim of raw FlashAttention kernel latency.
            fn=lambda: out.copy_(F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0))
        else:
            fn=lambda: ref.materialized_into(q,k,v,out)
        b,h,s,d=dims
        # Causal useful FLOPs: two matmuls, each 2*d FLOPs per valid pair.
        flops=4*b*h*d*(s*(s+1)//2)
        return fn,out,expected,flops,4*q.numel()*2,quant_error
    if op == "quant_gemm":
        if impl=="library":
            raise ValueError("No portable library baseline for the canonical packed format; use reference or student")
        m,n,k=dims
        a=rand((m,k),device,torch.float16,17);b=rand((k,n),device,torch.float16,23)
        qa=quantize_reference(a,variant);qb=quantize_reference(b.t().contiguous(),variant)
        canonical=PreparedQuantizedGemm(qa,qb)
        expected=quantized_gemm_reference(canonical)
        quant_error=error_metrics(expected,ref.gemm(a,b))
        out=torch.full_like(expected,float("nan"))
        if impl=="student":
            packed=student.prepare_quantized(qa,qb)
            fn=lambda: student.quant_gemm(packed,out,config=config)
            if quant_scope=="end_to_end":
                def fn():
                    p=student.prepare_quantized(quantize_reference(a,variant),
                         quantize_reference(b.t().contiguous(),variant))
                    student.quant_gemm(p,out,config=config)
        else:
            fn=lambda: out.copy_(quantized_gemm_reference(canonical))
            if quant_scope=="end_to_end":
                def fn():
                    p=PreparedQuantizedGemm(quantize_reference(a,variant),
                         quantize_reference(b.t().contiguous(),variant))
                    out.copy_(quantized_gemm_reference(p))
        payload=sum(t.numel()*t.element_size() for t in (qa.data,qa.scales,qb.data,qb.scales,out))
        return fn,out,expected,2*m*n*k,payload,quant_error
    raise ValueError(f"Unsupported operation: {op}")


def append_csv(path, row):
    if path.exists():
        with path.open(newline="") as f:
            if next(csv.reader(f),[]) != COLUMNS:
                raise ValueError(f"Existing CSV has a different schema: {path}")
    first=not path.exists()
    with path.open("a",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS)
        if first: writer.writeheader()
        writer.writerow(row)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("assignment",choices=ASSIGNMENTS)
    p.add_argument("--op")
    p.add_argument("--impl",choices=("reference","library","student"),default="student")
    p.add_argument("--variant")
    p.add_argument("--suite",choices=("smoke","full"),default="smoke")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--config",default="{}",help="JSON overrides of KernelConfig")
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--evict-mb",type=int,default=0)
    p.add_argument("--graph",action="store_true")
    p.add_argument("--quant-scope",choices=("prepacked","end_to_end"),default="prepacked")
    p.add_argument("--output",type=Path)
    args=p.parse_args(argv)
    if args.assignment=="a6":
        p.error("Use torchrun -m assignments.a6_moe.run for distributed tests/benchmarks")
    spec=ASSIGNMENTS[args.assignment];op=args.op or next(iter(spec["ops"]))
    if op not in spec["ops"]: p.error(f"Operations: {list(spec['ops'])}")
    variant=args.variant or spec["ops"][op][0]
    if variant not in spec["ops"][op]: p.error(f"Variants: {spec['ops'][op]}")
    try:
        config=json.loads(args.config); KernelConfig.parse(config)
    except (ValueError,TypeError) as exc: p.error(str(exc))
    if args.device=="cpu" and (args.impl!="reference" or args.suite!="smoke"):
        p.error("CPU is limited to --impl reference --suite smoke; it is not student GPU grading")
    if args.graph and (args.assignment!="a5" or args.device!="cuda"):
        p.error("--graph is only supported for A5 on CUDA")
    if args.warmup<1 or args.repeats<1 or args.evict_mb<0:
        p.error("warmup/repeats must be positive; evict-mb nonnegative")
    device=torch.device(args.device)
    if device.type=="cuda":
        require(spec["arch"] if args.impl=="student" else "cuda")
    torch.set_num_threads(min(torch.get_num_threads(),4))
    plan_path=Path(__file__).resolve().parents[1]/"assignments"/spec["module"]/"benchmark.json"
    dims_list=json.loads(plan_path.read_text())[args.suite][op]
    folder=args.output or Path("results")/args.assignment
    folder.mkdir(parents=True,exist_ok=True)
    run_id=uuid.uuid4().hex[:12]
    env=snapshot();env["arguments"]={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    env["run_id"]=run_id;env["status"]="started"
    env_path=folder/f"environment-{run_id}.json"
    env_path.write_text(json.dumps(env,indent=2,ensure_ascii=False)+"\n")
    try:
        with torch.no_grad(),strict_reference_precision():
            for dims in dims_list:
                fn,out,expected,flops,io_bytes,quant_err=prepare_case(args.assignment,op,dims,args.impl,variant,device,config,args.quant_scope)
                fn()
                atol,rtol=(2e-6,2e-5) if args.assignment=="a1" else (0.03,0.01)
                assert_output(out,expected,atol=atol,rtol=rtol)
                err=error_metrics(out,expected)
                peak=None
                if device.type=="cuda":
                    # Compile/cache before measuring workspace or timing.
                    fn();torch.cuda.synchronize(device)
                    before=torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                t=measure(fn,device=device,warmup=args.warmup,repeats=args.repeats,evict_mb=args.evict_mb,graph=args.graph)
                if device.type=="cuda":
                    peak=max(0,torch.cuda.max_memory_allocated(device)-before)
                gpu=env["devices"][torch.cuda.current_device()] if device.type=="cuda" else {}
                row=dict(zip(COLUMNS,[None]*len(COLUMNS)))
                row.update(run_id=run_id,assignment=args.assignment,op=op,impl=args.impl,variant=variant,
                    shape=json.dumps(dims),config=json.dumps(config,sort_keys=True),device=str(device),
                    gpu_name=gpu.get("name"),sm=json.dumps(gpu.get("compute_capability")),
                    scope=t.scope,cache_policy=t.cache_policy,
                    quant_scope=args.quant_scope if op=="quant_gemm" else "n/a",
                    warmup=args.warmup,repeats=args.repeats,p50_ms=t.p50_ms,
                    p10_ms=t.percentile(.1),p90_ms=t.percentile(.9),
                    tflops_estimate=flops/t.p50_ms/1e9 if device.type=="cuda" and t.p50_ms>0 else None,
                    logical_io_gbs_estimate=io_bytes/t.p50_ms/1e6 if device.type=="cuda" and t.p50_ms>0 else None,
                    peak_extra_allocated_bytes=peak,kernel_max_abs=err["max_abs"],kernel_relative_rmse=err["relative_rmse"],
                    quantization_max_abs=quant_err["max_abs"],quantization_relative_rmse=quant_err["relative_rmse"],
                    samples_ms=json.dumps(t.samples_ms))
                append_csv(folder/"bench.csv",row)
                print(f"{args.assignment} {op} {dims} {args.impl}/{variant}: {t.p50_ms:.4f} ms [{t.scope}]",flush=True)
        env["status"]="completed"
    except Exception as exc:
        env["status"]="failed";env["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env_path.write_text(json.dumps(env,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__": main()
