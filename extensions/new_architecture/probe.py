"""Read compiler targets; optionally compile generic code. No GPU claim."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.target and not re.fullmatch(r"sm_[0-9]+[af]?",args.target):p.error("Expected a literal nvcc target such as sm_100a")
    data={"nvcc":shutil.which("nvcc"),"target":args.target,"generic_compile": "not_tested",
          "feature_compile":"not_tested","hardware_correctness":"not_tested","hardware_performance":"not_tested"}
    if data["nvcc"]:
        for flag in ("--version","--list-gpu-code"):
            result=subprocess.run([data["nvcc"],flag],capture_output=True,text=True,timeout=30)
            data[flag]={"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}
        if args.target:
            with tempfile.TemporaryDirectory() as d:
                source=Path(d)/"probe.cu";binary=Path(d)/"probe.cubin"
                source.write_text('__global__ void probe(float* x) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<32) x[i]+=1.0f; }\n')
                result=subprocess.run([data["nvcc"],f"-arch={args.target}","-cubin",str(source),"-o",str(binary)],capture_output=True,text=True,timeout=120)
                data["generic_compile"]="passed" if result.returncode==0 else "failed"
                data["compiler_output"]={"stdout":result.stdout,"stderr":result.stderr,"returncode":result.returncode}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))
    return 1 if args.target and data["generic_compile"]!="passed" else 0


if __name__=="__main__":raise SystemExit(main())
