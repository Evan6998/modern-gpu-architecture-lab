"""Automated correctness gate. It never assigns architecture/performance points."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from mgpu.registry import ASSIGNMENTS


def junit_counts(path):
    if not path.exists():return {k:0 for k in ("tests","failures","errors","skipped")}
    root=ET.parse(path).getroot()
    suites=[root] if root.tag=="testsuite" else list(root.iter("testsuite"))
    return {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("assignment",choices=ASSIGNMENTS)
    p.add_argument("--cpu",action="store_true",help="Reference/harness check only; NOT assignment credit")
    p.add_argument("--variant",help="Select one pytest variant / one A6 implementation")
    p.add_argument("--nproc",type=int,default=2,choices=(2,4,8))
    p.add_argument("--output",type=Path)
    args=p.parse_args(argv)
    root=Path(__file__).resolve().parents[1]
    out=(args.output or root/"results"/args.assignment/"grade").resolve();out.mkdir(parents=True,exist_ok=True)
    commands=[]
    if args.assignment=="a6" and not args.cpu:
        variants=[args.variant] if args.variant else ["serial","overlap"]
        if any(v not in ("serial","overlap") for v in variants):p.error("A6 variants: serial, overlap")
        for v in variants:
            commands.append([sys.executable,"-m","torch.distributed.run","--standalone",f"--nproc_per_node={args.nproc}",
                "-m","assignments.a6_moe.run","--impl","student","--device","cuda","--variant",v,"--smoke","--output",str(out/v)])
    else:
        target=root/"assignments"/ASSIGNMENTS[args.assignment]["module"]
        cmd=[sys.executable,"-m","pytest",str(target),"-q",f"--junitxml={out/'junit.xml'}"]
        cmd += ["-m","not gpu and not distributed"] if args.cpu else ["-m","gpu","--run-gpu","--require-gpu","--strict-hardware"]
        if args.variant:cmd += ["-k",args.variant]
        commands.append(cmd)
    codes=[]
    for cmd in commands:
        print("Running:"," ".join(cmd),flush=True)
        try:codes.append(subprocess.run(cmd,cwd=root,timeout=600).returncode)
        except subprocess.TimeoutExpired:codes.append(124)
    counts=junit_counts(out/"junit.xml") if not (args.assignment=="a6" and not args.cpu) else None
    ok=all(code==0 for code in codes)
    if counts is not None:ok=ok and counts["tests"]>0 and counts["skipped"]==0
    status=("harness_only_not_assignment_credit" if args.cpu else "automated_correctness_passed_manual_review_required") if ok else "failed_or_incomplete"
    summary={"assignment":args.assignment,"status":status,"cpu_only":args.cpu,"exit_codes":codes,
             "counts":counts,"manual_review_required":["architecture evidence","performance/ablation","report"],
             "numeric_grade":None}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return 0 if ok else 1


if __name__=="__main__":raise SystemExit(main())
