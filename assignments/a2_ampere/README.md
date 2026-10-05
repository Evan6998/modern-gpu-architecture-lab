# A2 · Tensor Core 与异步搬运

**Volta/Turing 编程思想 → Ampere · CUDA C++**


## 任务

沿同一个 GEMM 依次实现 SIMT tiled、warp-level Tensor Core、`cp.async` pipeline 三版。
Tensor Core 版本使用 `mma.sync` 与 `ldmatrix`，画清 lane 持有的 fragments；
异步版本明确 prologue / steady state / drain，保留同步搬运版本作消融。[AMPERE][PTX]

## 接口与边界

```python
gemm(a, b, out, *, variant="simt", config=None)
# A[M,K], B[K,N], C[M,N]；输入/输出 FP16，FP32 累加
```

所有数据 row-major、contiguous、无梯度、out 不得 alias。A2 包括非整齐 M/N/K，
边缘 tile 可以使用你自己的 SIMT kernel，但不能调用库 GEMM 代替核心计算。
**实现位置：`csrc/kernels.cu`**。参考与库 baseline 分开；不可修改输入或返回新 tensor 代替 out。

## 公开测试

已知答案、零矩阵与非法接口在 CPU 检查；GPU 覆盖 1×1×1、37×53×29、不同 M/N/K、
非默认 stream、输入不变。async 额外测试 stages=1/2/3 与 K=32/96/288，并多次改变输入，
暴露 pipeline wraparound、遗漏尾部和陈旧结果问题。

```bash
python -m mgpu.grade a2 --variant simt
python -m mgpu.grade a2 --variant async
python -m mgpu.bench a2 --op gemm --variant async --suite full --config '{"stages":3}'
python -m mgpu.bench a2 --op gemm --impl library --suite full
```

## 实验与提交

固定 `(M,N,K)` 为 `(4096,4096,4096)`、`(128,4096,4096)`、`(4096,4096,128)`。
小范围比较 tile 与 1/2/3 stages；报告 SMEM、register、有效算力与 memory stall。
提交 MMA / async 指令证据；不能只用“用了 Tensor Core”解释全部收益。
在 H100 上跑的旧模型记为 **H100 / Ampere-style kernel**，不能标成 A100 性能。

## 提交与评分

填写本目录 `REPORT.md` 和 `submission.json`，附源码改动、测试结果、CSV 与必要的 trace/PTX/SASS。
每题 100 分：正确性 40、架构证据 30、测量消融 20、报告 10。
完整规则见 [GRADING.md](../../docs/GRADING.md)，资料标签见 [SOURCES.md](../../docs/SOURCES.md)。
