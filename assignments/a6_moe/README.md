# A6 · 多 GPU Mini-MoE

**单卡 → NVLink/NVSwitch → 网络跨域 · CUDA + NCCL / torch.distributed**


## 任务

固定 8 个 experts、top-1 routing、每 expert 一个 linear，只做 forward：
`permute -> dispatch A2A -> local expert GEMM -> return A2A -> unpermute`。
先 serial，再分块 overlap；复用 A2–A4 的 GEMM，不训练 router、不实现完整 MLP/backward。
通信可以使用 NCCL / torch.distributed；算法与 split-size/metadata 管理由你实现。[TORCH-DIST]

数学定义：`out[t] = x[t] @ W[routes[t]]`，其中 `W[e]` 是 expert e 的 `[D,H]` 权重，无 bias；
FP16 输入/输出、FP32 累加，与 A2 的 GEMM 约定相同。

**复用 GEMM 时注意 shape**：expert GEMM 的 M 是路由到该 expert 的 token 数，任意且可为 0；
smoke 用 D=64、H=32。只有 A2 的 GEMM 接受任意 shape。A3 / A4 的 kernel 要求 M、N 为 128 倍数
（且分别只在 SM90 / SM100 上运行），直接复用必须自己 pad M 和 N，pad 出的行列不能写回结果，
其 workspace 与拷贝开销要计入报告。正确性验收建议先用 A2 的 kernel 跑通。

## 接口与所有权

```python
moe(x, local_weights, routes, out, *, variant="serial", config=None, group=None)
# X[T,D] FP16, local_weights[8/world,D,H] FP16
# routes[T] int64, 0 <= expert < 8; out[T,H] FP16
```

rank r 持有 `[r*(8/world), (r+1)*(8/world))` 的 experts。支持 world=2/4/8。
每 rank 输入 token 数可不同，也可以为 0；每个实际 token 都必须返回原位置。
`routes` 合法性由数据生成器在计时外保证，不能在 hot path 每次 `.item()` 做同步。
空 rank 仍必须参与其他 rank 所需的 collectives。
**实现位置：`student.py`**，进程启动、测试数据、oracle 和 worst-rank 计时已提供。

## 公开测试与运行

uniform、90% 倾斜、all-to-one、不同 rank token 数（rank0=0）、所有 rank 全空。
改变输入后复用 output，检查顺序、输入不变和陈旧结果。
reference baseline 会 gather 全部权重，只用于验证 harness；学生实现不得照搬。

```bash
# 无 GPU 也可以运行，检查真正的双进程基础设施，不是学生算法评分
make dist-smoke

# 同机 GPU 验收，默认检查全部五种 routing case
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m assignments.a6_moe.run --impl student --device cuda --variant serial --smoke
python -m mgpu.grade a6 --nproc 2  # 两个 variant 都检查

# 8 卡性能，先完成正确性；不会自动提交集群作业
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m assignments.a6_moe.run --impl student --variant overlap --bench \
  --tokens 4096 --d-model 4096 --d-out 4096 --case uniform --case skew
```

两节点每节点 4 ranks，总 world=8（保持 8 experts 不变）。在已分配的两节点分别运行：

```bash
# 两端设置相同的 MASTER_ADDR / MASTER_PORT，以及各自 NODE_RANK=0 或 1
python -m torch.distributed.run --nnodes=2 --nproc_per_node=4 \
  --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m assignments.a6_moe.run --impl student --variant overlap --bench \
  --tokens 4096 --d-model 4096 --d-out 4096 --case uniform --case skew
```

## 实验与提交

比较 2 ranks、8 ranks、2×4 ranks；把真实 `nvidia-smi topo -m` 和网络配置附上，
不要默认 node boundary 就是 NVLink / 网络边界。给 dispatch/compute/return 标 NVTX ranges。
证明 overlap 后**总时间**改善，或解释为什么没有改善；不能只凭 trace 上出现并发就判断加速。
计时逐轮取最慢 rank，额外保存每个 rank 的样本；所有 rank 的 JSON 都要提交。

## 提交与评分

填写本目录 `REPORT.md` 和 `submission.json`，附源码改动、测试结果、CSV 与必要的 trace/PTX/SASS。
每题 100 分：正确性 40、架构证据 30、测量消融 20、报告 10。
完整规则见 [GRADING.md](../../docs/GRADING.md)，资料标签见 [SOURCES.md](../../docs/SOURCES.md)。
