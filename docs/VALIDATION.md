# 验证记录

这里记录的是框架质量，不是学生 GPU kernel 的完成成绩。第一部分是初始化时的记录，
第二部分是 2026-10-05 审题修订后的复验，这两次都没有 NVIDIA GPU；第三部分是之后在两台 GPU 机器上做的首检。

## 初始化（Linux x86_64，PyTorch 2.10.0+cpu）

本次会话工作区实测。

| 检查 | 实际结果 |
|---|---|
| `python -m pytest -q -m 'not gpu and not distributed'` | **116 passed，98 GPU tests deselected** |
| 默认 `python -m pytest -q` | **116 passed，98 GPU tests skipped**；跳过明确不计作业成绩 |
| 两个真实进程，CPU/Gloo | uniform、skew、all_to_one、uneven、empty 全部通过 |
| 分布式计时管道 | 逐轮最慢 rank 汇总、每 rank JSON、CSV 输出已运行 |
| A1–A5 reference benchmark smoke | 入口、正确性检查、CPU 计时标记与 CSV 输出通过 |
| A4 MXFP4 end-to-end reference smoke | 量化、prepare、oracle 与两类误差输出通过 |
| `mgpu.grade a1 --cpu` | 明确返回 harness-only / no assignment credit |
| 无 CUDA 时 `mgpu.grade a1` | 正确拒绝，非零退出码 1 |
| editable install | `pip install --no-deps --no-build-isolation -e .` 成功 |
| Python / shell | compileall 与全部 shell scripts 的 `bash -n` 通过 |

实际 Python 为 3.13.5，PyTorch 为 **2.10.0+cpu**；没有 NVIDIA GPU、CUDA runtime 或 nvcc。
机器详情与测试证据在 [validation/](validation/) 中。

**未验证**：CUDA C++ 编译、学生 GPU kernel 正确性/性能、CuTe/Triton GPU 运行、NCCL、跨节点运行、
Nsight / Compute Sanitizer、Rubin 指定 feature 编译及真机行为。
GPU 相关测试和命令已准备，但必须在匹配硬件上实际运行后才能声称通过。
A1–A6 核心计算仍是有意保留的 TODO。没有将参考实现接入 student 路径，也没有生成 GPU 性能数字。

CPU benchmark 的零散时间仅用来检查计时/日志代码，不适用于硬件性能比较，因此未作为课程 benchmark 成绩附入。
GitHub Actions workflow 已写好，但本会话没有把项目推送到 GitHub 或实际运行云端 CI。
[validation/](validation/) 下的文件都是这次初始化运行的原始证据，测试数量与修订后不同。

## 审题修订后复验（macOS 26.5 arm64，Python 3.11.14，PyTorch 2.14.1，无 CUDA）

| 检查 | 实际结果 |
|---|---|
| `make check` | 通过 |
| `make cpu` | **128 passed，121 GPU tests deselected** |
| 默认 `python -m pytest -q` | **128 passed，121 skipped** |
| `make dist-smoke` | 五个 routing case 全部通过 |
| A1–A5 reference benchmark smoke（CPU） | 九个 op / 格式组合全部通过，含 A4 MXFP4 end-to-end |
| `mgpu.grade a1 --cpu` / 无 CUDA 的 `mgpu.grade a1` | harness-only / 正确拒绝 |
| `bindings.cpp`（A1、A2） | 用 host clang 对 PyTorch 2.14.1 头文件做 `-std=c++20 -fsyntax-only`，通过 |

修订中确认并修复的 harness 问题：

| 问题 | 怎么确认的 | 修复 |
|---|---|---|
| `mgpu/extension.py` 强制 `-std=c++17`，而 PyTorch ≥ 2.12 的 extension 需要 C++20；`pyproject` 只约束 `torch>=2.6`，新环境里 A1 / A2 无法编译 | 2.14.1 头文件在 C++17 下 `#error`；比对各版本 `cpp_extension.py` 的默认标准 | 不再传 `-std` |
| `make dist-smoke` 在主机名不可解析的机器上挂起到超时 | macOS 上复现，c10d 反复解析主机 FQDN 失败 | rendezvous 固定到 `127.0.0.1` |
| A5 测试数据的 logits 标准差约 0.06，attention 近似 prefix average；漏掉 online softmax 重缩放的实现，最坏误差只有容差的 0.77–1.17 倍 | CPU 上用 FP32 仿真正确实现与三种错误实现，逐一对照 oracle | 增加 `peaked` 数据（重缩放错误的误差升到容差的 60 倍以上，正确实现仍为 0.044 倍）；benchmark 数据同步 |
| benchmark 的正确性容差比验收宽（attention 0.03 对 0.003，transpose / cluster_copy 0.03 对精确相等） | 读代码 | 统一到 `mgpu/checks.py` 的一张表 |
| `peak_extra_allocated_bytes` 把 `--evict-mb` 的 buffer 算进算子 workspace | 读代码；用假的 allocator 驱动 `measure()` 的 CUDA 分支 | 基线移到 buffer 分配之后 |
| `reference` / `library` 的 CSV 行带着学生 variant 名 | CPU 实跑 | 记为 `n/a` |
| A1 的 `--config` 被静默忽略，但仍写进 CSV | 读代码 | `tile_m` / `warps` 传入 kernel，其余 key 报错 |
| `mgpu.grade` 固定 600 秒超时，首次 JIT / nvcc 编译可能被判成失败；旧 `junit.xml` 可能被重复汇总 | 读代码 | `--timeout`（默认 1800）；运行前删除旧文件 |
| `make bench A=a2` 因默认 `OP=transpose` 直接报错 | 实跑 | `OP` 默认为空 |
| A3 GEMM 公开测试只有单个 128×128 tile；A4 量化测试只有 128×128×64 | 读代码 | 增加 M≠N 的多 tile shape |
| A4 MXFP4 合约写 K%32，而 `kind::mxf4` 一条 MMA 消耗 K=64 | 查阅资料 | 合约收紧为 K%64，题面写明 |

写下上表时，这些修订都还没有在 GPU 上验证过，尤其是：去掉 `-std` 后的 nvcc 实际编译、A1 extension 的新签名
（`kernels.cu` 只在 host 侧检查了 `bindings.cpp`）、`measure()` 的 CUDA 分支、新增的 23 个 GPU 测试、
以及 library attention baseline 在放大后的 Q 上能否通过收紧的容差（CPU 上的 FP16 仿真误差约为容差的 0.26 倍）。
其中一部分在下一节补上了。

## GPU 首检（2026-10-06）

用 `scripts/gpu_smoke.sh` 在两台机器上各跑了一遍。验证对象仍然是框架和 starter，没有任何学生 kernel。

| 项目 | 1× NVIDIA A10（SM 8.6） | 1× NVIDIA GB200（SM 10.0） |
|---|---|---|
| 环境 | x86_64，Ubuntu 22.04，Python 3.10.12，PyTorch 2.7.0（CUDA 12.8），nvcc 12.8.93，g++ 11.4.0，driver 570.148.08 | aarch64，Debian，Python 3.12.4，PyTorch 2.9.1（CUDA 13.1），nvcc 13.1.115，g++ 12.2.0，driver 580.126.20，Triton 3.6.0，CuTe DSL 4.4.2 |
| `make doctor` 识别的 family | cuda、ampere | cuda、ampere、blackwell |
| `make check` / `make cpu` | 通过 | 通过（128 passed，121 deselected） |
| A1 / A2 starter | 编译成功（`sm_86`），停在 `TODO` | 编译成功（`sm_100`），停在 `TODO` |
| library baseline 通过 benchmark 正确性检查 | A1 transpose / softmax、A2 gemm、A5 attention 全部通过 | 同左 |
| `--evict-mb 64` 时的 `peak_extra_allocated_bytes` | 0 | 0 |
| `ncu` 读取 GPU 性能计数器 | 未验证（镜像没有 Nsight Compute） | 通过，不需要 sudo |
| Triton、CuTe DSL（含 tcgen05 子模块）import | 未检查 | 通过 |

A10 那台的 PyTorch 是系统包，不带 pybind11 头文件，starter 第一次编译因此失败；
`mgpu/extension.py` 里从 `pybind11` 模块取 include 路径的兜底就是这时加的，加上后两台都编译通过。

这次确认了上一节里原先没验证的几项：去掉 `-std` 后的 nvcc 实际编译（PyTorch 2.7 和 2.9，都是 C++17 路径）、
A1 extension 的新签名、`measure()` 的 CUDA 分支和 eviction buffer 的统计、
library attention baseline 在放大后的 Q 上通过收紧的容差。

**仍未验证**：任何学生 kernel 的正确性和性能（121 个 GPU 测试在 starter 上按设计失败）、
PyTorch ≥ 2.12 的 C++20 编译路径、A3（需要 SM 9.0）、A6 的 NCCL / 多卡 / 跨节点、
`compute-sanitizer` 与 `nsys` 的实际采集、Rubin。原始日志没有收进仓库。
