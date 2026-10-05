"""Small reusable GPU-test helpers. Not a replacement for kernel implementation."""
import torch
from mgpu.bench import rand
from mgpu.checks import assert_output
from mgpu.registry import module


def exercise_gemm(aid, dims, variant, device, config=None, repetitions=1):
    m,n,k=dims
    ref,student=module(aid,"reference"),module(aid,"student")
    stream=torch.cuda.Stream(device=device)
    for repeat in range(repetitions):
        with torch.cuda.stream(stream):
            a=rand((m,k),device,torch.float16,17+repeat)
            b=rand((k,n),device,torch.float16,31+repeat)
            a0,b0=a.clone(),b.clone()
            expected=ref.gemm(a,b)
            out=torch.full((m,n),float("nan"),dtype=torch.float16,device=device)
            student.gemm(a,b,out,variant=variant,config=config)
            # Depend on the result in the SAME non-default stream.
            observed=out.clone()
        stream.synchronize()
        assert_output(observed,expected)
        torch.testing.assert_close(a,a0,atol=0,rtol=0)
        torch.testing.assert_close(b,b0,atol=0,rtol=0)
