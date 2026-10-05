"""torchrun entry point for correctness and worst-rank end-to-end timing."""
import argparse
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
import uuid
import torch
import torch.distributed as dist
from mgpu.checks import assert_output
from mgpu.config import KernelConfig
from mgpu.doctor import snapshot
from mgpu.precision import strict_reference_precision
from . import reference, student

CASES=("uniform","skew","all_to_one","uneven","empty")


def make_data(case, rank, world, tokens, d_model, d_out, device):
    if case not in CASES: raise ValueError("Unknown case")
    n=(0 if rank==0 else tokens+rank*7) if case=="uneven" else tokens
    if case=="empty": n=0
    g=torch.Generator().manual_seed(104729+rank)
    x=(torch.randn((n,d_model),generator=g)*.25).half()
    if n: x[:,0]=torch.arange(n).float()/max(n,1)
    routes=(torch.arange(n)+rank)%8
    if case=="skew":
        routes[:int(n*.9)]=0
        if n: routes=routes[torch.randperm(n,generator=g)]
    elif case=="all_to_one": routes.zero_()
    weights=(torch.randn((8,d_model,d_out),generator=torch.Generator().manual_seed(65537))*.25).half()
    per_rank=8//world
    # Clone ensures the student's local shard has no full-weight backing storage.
    local=weights[rank*per_rank:(rank+1)*per_rank].clone().to(device)
    return x.to(device),local,routes.to(device),weights.to(device)


def synchronize(device):
    if device.type=="cuda": torch.cuda.synchronize(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--impl",choices=("reference","student"),default="student")
    p.add_argument("--variant",choices=("serial","overlap"),default="serial")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--bench",action="store_true")
    p.add_argument("--tokens",type=int,default=257)
    p.add_argument("--d-model",type=int,default=128)
    p.add_argument("--d-out",type=int,default=64)
    p.add_argument("--case",choices=CASES,action="append")
    p.add_argument("--config",default="{}")
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=10)
    p.add_argument("--timeout",type=int,default=120)
    p.add_argument("--output",type=Path,default=Path("results/a6"))
    args=p.parse_args()
    if args.smoke: args.tokens,args.d_model,args.d_out=33,64,32
    if args.tokens<0 or min(args.d_model,args.d_out,args.warmup,args.repeats,args.timeout)<1:
        p.error("Invalid dimensions or timing/timeout arguments")
    if "RANK" not in os.environ: p.error("Launch with torchrun, not plain python")
    if args.device=="cpu" and args.impl!="reference": p.error("CPU/Gloo only validates the reference harness")
    try:
        config=json.loads(args.config);KernelConfig.parse(config)
    except (ValueError,TypeError) as exc:p.error(str(exc))
    rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"])
    if world not in (1,2,4,8):p.error("Eight experts require world size in {1,2,4,8}")
    if args.device=="cuda":
        if not torch.cuda.is_available():p.error("CUDA unavailable")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device=torch.device("cuda",int(os.environ["LOCAL_RANK"]));backend="nccl"
    else:device=torch.device("cpu");backend="gloo"
    torch.set_num_threads(min(torch.get_num_threads(),2))
    dist.init_process_group(backend,timeout=timedelta(seconds=args.timeout))
    # Same numeric run id on all ranks without object collectives on CUDA.
    run_tensor=torch.tensor([int(uuid.uuid4().hex[:12],16) if rank==0 else 0],dtype=torch.int64,device=device)
    dist.broadcast(run_tensor,src=0);run_id=f"{int(run_tensor.item()):012x}"
    args.output.mkdir(parents=True,exist_ok=True)
    report={"run_id":run_id,"rank":rank,"world_size":world,"backend":backend,"impl":args.impl,
            "variant":args.variant,"cases":[],"status":"started",
            "assignment_credit":False,"environment":snapshot(),"config":config}
    path=args.output/f"run-{run_id}-rank{rank}.json"
    try:
        with torch.no_grad(),strict_reference_precision():
            for case in args.case or CASES:
                x,weights,routes,all_weights=make_data(case,rank,world,args.tokens,args.d_model,args.d_out,device)
                expected=reference.moe(x,all_weights,routes)
                del all_weights
                out=torch.full_like(expected,float("nan"))
                x_copy=x.clone();w_copy=weights.clone();r_copy=routes.clone()
                def invoke():
                    if args.impl=="student":student.moe(x,weights,routes,out,variant=args.variant,config=config)
                    else:reference.distributed_into(x,weights,routes,out)
                invoke();synchronize(device)
                assert_output(out,expected)
                for original,current in ((x_copy,x),(w_copy,weights),(r_copy,routes)):
                    torch.testing.assert_close(current,original,atol=0,rtol=0)
                # Reuse buffers with changed data; stale-output caching must fail.
                x.mul_(.5)
                expected.mul_(.5)
                out.fill_(float("nan"));invoke();synchronize(device)
                assert_output(out,expected)
                samples=[];local_samples=[]
                if args.bench and case!="empty":
                    for _ in range(args.warmup):invoke()
                    synchronize(device)
                    for _ in range(args.repeats):
                        dist.barrier();synchronize(device)
                        start=perf_counter();invoke();synchronize(device)
                        elapsed=(perf_counter()-start)*1000
                        local_samples.append(elapsed)
                        worst=torch.tensor([elapsed],device=device,dtype=torch.float64)
                        # This reduction is deliberately OUTSIDE the timed region.
                        dist.all_reduce(worst,op=dist.ReduceOp.MAX)
                        samples.append(float(worst.item()))
                report["cases"].append({"case":case,"local_tokens":x.shape[0],"d_model":args.d_model,"d_out":args.d_out,
                    "correct":True,"local_samples_ms":local_samples,"worst_rank_samples_ms":samples,
                    "worst_rank_p50_ms":median(samples) if samples else None})
                if rank==0:print(f"PASS {case}: {args.impl}/{args.variant}, worst-rank p50={median(samples) if samples else 'not timed'} ms",flush=True)
            report["status"]="automated_correctness_passed" if args.impl=="student" else "reference_harness_passed_no_assignment_credit"
            # Architecture, dispatch algorithm and performance evidence are manual.
            if rank==0 and args.bench:
                csv_path=args.output/f"bench-{run_id}.csv"
                with csv_path.open("w",newline="") as f:
                    writer=csv.DictWriter(f,fieldnames=["run_id","case","impl","variant","world_size","worst_rank_p50_ms","worst_rank_samples_ms"])
                    writer.writeheader()
                    for row in report["cases"]:
                        writer.writerow({"run_id":run_id,"case":row["case"],"impl":args.impl,"variant":args.variant,
                            "world_size":world,"worst_rank_p50_ms":row["worst_rank_p50_ms"],
                            "worst_rank_samples_ms":json.dumps(row["worst_rank_samples_ms"])})
    except Exception as exc:
        report["status"]="failed";report["error"]=f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        dist.destroy_process_group()


if __name__=="__main__":main()
