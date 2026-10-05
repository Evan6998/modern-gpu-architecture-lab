# Project instructions

This is a university-style assignment starter kit, not a finished kernel library.
Preserve that distinction. Do not solve every assignment unless the user explicitly
asks for solutions. Work on one requested assignment at a time.

- Read that assignment's README and public API before editing.
- Keep reference implementations separate; never make student kernels fall back to them.
- Do not weaken tests or tolerances to obtain a pass. Document genuine harness bugs.
- CPU tests validate infrastructure only. No CUDA hardware means no GPU validation claim.
- Keep A2/A3/A4 GEMM inputs row-major A[M,K], B[K,N], output FP16, accumulation FP32.
- Respect the current PyTorch stream; do not hide synchronization in kernel launchers.
- Run `make check` and `make cpu` after harness changes; GPU tests need explicit opt-in.
- Do not install drivers, change clocks, reserve cluster nodes, or contact remote
  machines without a separate user request. No credentials belong in this repo.
- Record dependencies, benchmark commands, traces and architecture-specific evidence.
- Do not assume newer compute capabilities support every older architecture-specific ISA.
