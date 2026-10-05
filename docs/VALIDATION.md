# 初始化验证记录

本次会话工作区实测；这里记录的是框架初始化质量，不是学生 GPU kernel 的完成成绩。

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
