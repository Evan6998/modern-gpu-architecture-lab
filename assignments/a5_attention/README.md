# A5 · IO-aware Attention

**现代 kernel compiler 与融合 · Triton**


## 任务

实现 tiled causal attention forward，将 QKᵀ、online softmax、PV 融合。
不得将完整 `S×S` scores 或 probabilities 写回 global memory。
本题采用 Triton，把前题的硬件模型映射到 compiler 生成的代码。[TRITON-ATTN]

## 接口与边界

```python
attention(q, k, v, out, *, variant="fused", config=None)
# 所有 tensor: [B,H,S,128] FP16 contiguous，out 同形
```

固定 head_dim=128、causal、FP16 输入/输出、FP32 中间累加。
不做 dropout、GQA、backward。必须支持非整齐 S。
**实现位置：`kernels.py`**，已有 Triton kernel 签名和 launcher 接口；先实现 kernel 再启用 launcher。
运行之前安装匹配设备的 Triton；CPU reference 测试不需要 import Triton。

## 公开测试

S=1/17/129/256/1024；零 logits 应得到 causal prefix average；随机数据、future-token invariance、
输入不变和 PyTorch allocator 可观察的 workspace 检查。
workspace test 不是完整的 adversarial 内存审计，仍需 profiler 检查实际 HBM traffic。

```bash
python -m mgpu.grade a5
python -m mgpu.bench a5 --op attention --suite full
python -m mgpu.bench a5 --op attention --impl reference --suite full
python -m mgpu.bench a5 --op attention --impl library --suite full
python -m mgpu.bench a5 --op attention --graph --suite full
```

## 实验与提交

固定 batch=1、heads=16、S=256/2048/8192。只扫描少量 tile/warps/stages。
reference 路径会物化大矩阵，8192 case 显存占用明显更大，先检查可用资源。
分别解释减少中间数据、减少 launch、流水线变化的收益；短序列重点比较 Graph replay。
附编译器实际选择的 matrix/copy 指令证据，不要仅说“用 Triton 所以自动优化了”。

## 提交与评分

填写本目录 `REPORT.md` 和 `submission.json`，附源码改动、测试结果、CSV 与必要的 trace/PTX/SASS。
每题 100 分：正确性 40、架构证据 30、测量消融 20、报告 10。
完整规则见 [GRADING.md](../../docs/GRADING.md)，资料标签见 [SOURCES.md](../../docs/SOURCES.md)。
