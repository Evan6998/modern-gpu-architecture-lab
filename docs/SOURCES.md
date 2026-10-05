# 官方资料与 API 来源

查阅日期：2026-10-05。以下 live 文档会继续更新；实验时记录实际安装版本，不把网页版本号当作已安装版本。
本项目代码为教学脚手架，没有从这些资料复制完整 kernel 解答。

- **[CUDA-BEST]** [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：coalescing、shared-memory banks、计时、有效带宽与 profiler 口径。
- **[AMPERE]** [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)：异步 global-to-shared copy、Tensor Core 模型的代际变化。
- **[HOPPER]** [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：TMA、thread-block cluster、DSM、warp specialization。
- **[BLACKWELL]** [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)：SM100 / SM120 区分与内存、cluster 等硬件背景。
- **[TCGEN05]** [CuTe tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)：TMEM、CTA-pair、block-scaled MMA 的原语和数据流。
- **[CUTLASS]** [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)：安装、构建及 DSL 使用入口。
- **[PTX]** [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：mma、wgmma、cp.async、tcgen05、数值格式与目标架构限制。代码中的 MXFP4 是 E2M1 + E8M0 / 32-element scaling；量化时的 scale 选择策略属于本作业约定。
- **[TRITON-ATTN]** [Triton Fused Attention Tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：Triton attention 教学与相关实现思路。
- **[TORCH-EXT]** [PyTorch C++/CUDA Extension API](https://docs.pytorch.org/docs/stable/cpp_extension.html)：lazy extension load、CUDA toolkit 和编译依赖。
- **[TORCH-DIST]** [PyTorch Distributed API](https://docs.pytorch.org/docs/stable/distributed.html)：进程组、NCCL、collectives、异步操作和通信同步语义。
- **[CUDA-CURRENT]** [CUDA Toolkit 文档首页](https://docs.nvidia.com/cuda/)：此次查阅的首页列出 CUDA 13.4 与 Rubin preview；这不是对某台机器硬件可用性的验证。

资料阅读只取当前 assignment 所需段落，不要求通读整份规范。
