# Modern GPU Architecture Lab

**六个递进作业 · starter code + public tests + benchmark + rubric**

面向已有 ML systems 理论基础、希望实际写 kernel / profile / 调优的学习者。
这是课程初始化项目，不是六题的答案。核心实现刻意留有 `TODO` / `NotImplementedError`；
参考实现、数据生成、CPU 检查、计时和验收框架已提供。

## 路线

| 顺序 | 作业 | 主线 | 实现入口 |
|---|---|---|---|
| 1 | [SIMT 与内存](assignments/a1_memory/README.md) | transpose、softmax、coalescing、banks、显式同步 | `a1_memory/csrc/kernels.cu` |
| 2 | [Tensor Core / Ampere](assignments/a2_ampere/README.md) | SIMT GEMM → MMA → cp.async | `a2_ampere/csrc/kernels.cu` |
| 3 | [Hopper](assignments/a3_hopper/README.md) | TMA、WGMMA、warp specialization、cluster | `a3_hopper/kernels.py` |
| 4 | [Blackwell](assignments/a4_blackwell/README.md) | tcgen05、TMEM、2-SM、persistent、FP8/MXFP4 | `a4_blackwell/kernels.py` |
| 5 | [Attention](assignments/a5_attention/README.md) | Triton、online softmax、IO-aware fusion、Graph | `a5_attention/kernels.py` |
| 6 | [Mini-MoE](assignments/a6_moe/README.md) | A2A、expert GEMM、通信计算重叠 | `a6_moe/student.py` |

A2–A4 共用 `A[M,K] @ B[K,N] -> C[M,N]` 接口和三组主 benchmark。
“过去到最新”指关键计算/同步/存储编程模型的演进，不是每代产品或图形渲染功能的穷举。
A2 在支持的较新 GPU 上练习早期 Tensor Core 编程思想，不等于测出了 V100 的性能。

最新架构单独放在 [Rubin / 新架构迁移附录](extensions/new_architecture/README.md)：
官方 CUDA 文档目前列出 Rubin **preview**；预览支持、编译成功和硬件性能实测是三件不同的事。
不把未确认可用的硬件设为六题的默认依赖。资料见 [SOURCES.md](docs/SOURCES.md)。

## 从这里开始

在解压后的本目录运行。先装适合当前机器的 PyTorch；Linux GPU 机器上的 PyTorch wheel、
CUDA toolkit、驱动和 GPU 架构必须匹配，详见 [环境与硬件](docs/HARDWARE.md)。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

make doctor       # 检查环境，不修改驱动、频率或系统设置
make cpu          # 参考实现 + 测试框架；无需 NVIDIA GPU
make dist-smoke   # 两个真实进程，用 CPU/Gloo 检查分布式脚手架
```

开始 A1：

```bash
# 先读题，再编辑这个文件
# assignments/a1_memory/csrc/kernels.cu

# NVIDIA GPU 上，starter 未实现时这里应失败，而不是显示通过
make grade A=a1

# 初期也可只测一个 variant
python -m mgpu.grade a1 --variant naive

# 正确性通过后再测性能
python -m mgpu.bench a1 --op transpose --variant naive --impl student --suite full
python -m mgpu.bench a1 --op transpose --impl library --suite full
```

没有 GPU 的机器也能验证数据和 benchmark 管道：

```bash
python -m mgpu.bench a1 --op transpose --impl reference --device cpu --suite smoke
```

这里产生的是 **CPU reference wall time**，不是 GPU 性能。不要用它填写 GPU 作业成绩。

## 项目结构

```text
modern-gpu-architecture-lab/
├── assignments/a1_memory/ ... a6_moe/
│   ├── README.md             题目、接口、步骤、边界、验收命令
│   ├── student.py            学生实现入口
│   ├── reference.py          可运行的数值 oracle / baseline
│   ├── kernels.py 或 csrc/   kernel TODO
│   ├── tests/                CPU oracle + GPU public tests
│   ├── benchmark.json        smoke / full 工作负载
│   ├── REPORT.md             一页报告模板
│   └── submission.json       提交材料索引
├── mgpu/                     通用验证、计时、量化、评分入口
├── tests/                    harness 自测
├── docs/                     环境、测量规则、评分、来源、验证记录
├── extensions/new_architecture/  编译目标探测与迁移记录
├── scripts/                  profiling、sanitizer、环境锁定
├── .github/workflows/cpu.yml CPU CI
├── Makefile
└── AGENTS.md                 给代码助手的项目约束
```

## 什么算完成

每题提交实现、测试结果、`bench.csv`、架构使用证据和一页 `REPORT.md`。
自动测试只检查可观测的正确性，**不能自动证明**代码用了 TMA、WGMMA 或 FP4 Tensor Core。
这些需要 PTX/SASS、Nsight 和源码审查；详见 [评分规则](docs/GRADING.md)。

默认 `pytest` 会跳过 GPU tests。跳过不是通过。正式 `mgpu.grade` 会拒绝无 GPU、错误架构和零测试验收。
当前项目的实际验证范围请读 [VALIDATION.md](docs/VALIDATION.md)，不要把框架验证当成 kernel 已完成。
