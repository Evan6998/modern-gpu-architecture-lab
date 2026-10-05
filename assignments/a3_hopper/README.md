# A3 · Hopper 原生流水线

**Hopper · CuTe DSL 或 CUDA C++**


## 任务

沿用 A2 的 GEMM 接口，新增 `tma_wgmma`。producer 负责 TMA，consumer warpgroup 负责 WGMMA；
自己组织循环缓冲区、barrier、phase 与 buffer ownership。允许使用 CuTe descriptor/layout/同步原语，
不能把完整官方 GEMM 包一层就作为答案。[HOPPER][PTX]

另做一个固定双 CTA cluster 小题，三个 variant 输出完全相同：duplicate 重复加载，
multicast 用 TMA multicast，dsm 由 CTA1 读取 CTA0 的 shared memory。
不要让提供 DSM 数据的 CTA 在另一个 CTA 读完前退出。

## 接口与边界

```python
gemm(a, b, out, *, variant="tma_wgmma", config=None)
cluster_copy(x, out, *, variant="duplicate", config=None)
# cluster: X[128,128] FP32 -> out[2,128,128]，每个 out[i] == X
```

GEMM 仍是 FP16 / FP32 accumulate / FP16，**本题限定 M,N 为 128 倍数、K 为 32 倍数**。
这是控制工程量的明确限制，不要求写完整通用 GEMM。
**实现位置：`kernels.py`**；可替换其内部为 CUDA C++ backend，但保持 launcher 签名。
`pipeline_model.py` 是给定的 host ownership 模型，不等于硬件 barrier 的初始化答案。

`config` 读 `tile_m / tile_n / tile_k / stages / producer_warps / consumer_warpgroups`。公开测试用默认值
（tile 128×128×32）和 `stages=1/2/3`，这些必须可用；不支持的取值抛 `ValueError`，不要静默换成别的值。

## 公开测试

stages=1/2/3、不同 K、4 轮改变输入、非默认 stream、输入未修改；另有 M≠N 的多 tile shape
（256×128、128×384、384×256），单个 128×128 tile 测不出 grid 下标和 A/B 维度写反；cluster 三版重复随机输入。
精确 Hopper gate：SM90，不能默认把 WGMMA 当成未来所有 GPU 都支持的接口。

```bash
python -m mgpu.grade a3
python -m mgpu.bench a3 --op gemm --suite full --config '{"stages":2}'
python -m mgpu.bench a2 --op gemm --variant async --suite full  # 同一台 H100 上比较
python -m mgpu.bench a3 --op cluster_copy --variant multicast
```

## 实验与提交

画一张 buffer 生命周期时序图：谁写、谁读、何时 wait、何时允许复用。
比较 stage 数量及 producer/consumer 数量，证明搬运和计算的重叠，而不是仅提交函数名。
cluster 小题只需要固定 tile，不扩成第二套通用 cluster GEMM。

## 提交与评分

填写本目录 `REPORT.md` 和 `submission.json`，附源码改动、测试结果、CSV 与必要的 trace/PTX/SASS。
每题 100 分：正确性 40、架构证据 30、测量消融 20、报告 10。
完整规则见 [GRADING.md](../../docs/GRADING.md)，资料标签见 [SOURCES.md](../../docs/SOURCES.md)。
