# 测量约定

## 一套入口

```bash
python -m mgpu.bench a2 --op gemm --impl student --variant async --suite full --config '{"stages":3}'
python -m mgpu.bench a2 --op gemm --impl library --suite full
```

每次运行产生一个 run ID、独立的 environment JSON，并把每个 shape 的结果追加到 `bench.csv`。
已有 CSV 的 schema 不匹配会拒绝写入；不同架构 / 配置不能在分析时无条件混为一组。
`reference` / `library` 不经过学生 variant，这些行的 `variant` 记为 `n/a`；`quant_gemm` 例外，
它的 variant 是数据格式（fp8 / mxfp4），对所有实现都有意义。

## `--config`

`--config` 是 `KernelConfig` 的 JSON 覆盖，原样记入 CSV。各题只读其中一部分：

| 字段 | 默认 | 谁在用 |
|---|---:|---|
| `tile_m` / `tile_n` | 128 | A2–A4 输出 tile；A5 的 `BLOCK_M` / `BLOCK_N`；A1 transpose 只读 `tile_m`，且 A1 默认 32 |
| `tile_k` | 32 | A2–A4 每个 stage 的 K 步长 |
| `stages` | 2 | A2 `cp.async`、A3 / A4 ring buffer；A5 传给 Triton `num_stages` |
| `warps` | 4 | A1 softmax 每 block 的 warp 数、A2、A5 `num_warps` |
| `producer_warps` / `consumer_warpgroups` | 1 | A3 / A4 |
| `chunk_size` | 256 | A6 overlap 的分块 token 数 |

两条规则：公开测试用到的取值（默认值，以及 A2 / A3 的 `stages=1/2/3`）必须可用；
其余取值要么真的生效，要么显式报错。静默忽略或替换会让 CSV 里的消融标签与实际执行的 kernel 对不上。
A1 的 host 入口会直接拒绝它不读的字段；A2–A5 由你的 dispatcher / launcher 负责。

## 口径

`reference` 是数值 oracle，通常故意低效。`library` 是 PyTorch/library operator baseline。
`student` 才是作业实现。GPU 上先核对正确性，再预热、计时；编译和输入构造不计入 steady-state。
计时前的正确性检查与公开测试共用 `mgpu/checks.py` 里的同一张容差表，
不会出现“benchmark 记了一行、验收却不通过”的结果。
正式性能用 GPU event 的中位数，同时保留 p10、p90 和每次样本；CPU reference wall time 单独标记。
计时测的是 operator，而不是凭空保证某一条 kernel 的无 launch 开销时长。

所有单卡实现写入预分配 out，必须使用 PyTorch 当前 stream。使用内部 stream 的实现必须显式 join。
A6 使用 host wall time + device synchronize，逐轮取最慢 rank；barrier 和 max-reduction 放在计时区外。
不要用单个最快 rank，也不要把 max(median(rank)) 混同于 median(max(rank per iteration))。

`library` attention 包含框架输出 allocation 和 copy；`reference` attention 包含完整 score/probability
矩阵。CSV 不把这些路径冒充为同一低层 kernel 的比较。GEMM 使用相同 FP16 输入，关闭 reduced-precision
reduction 和 TF32 reference 路径，CPU oracle 使用 FP32 accumulation。[TORCH-EXT][TRITON-ATTN]

## Cache / bandwidth

默认 `warm_reused_inputs`。`--evict-mb N` 在计时之前写入一个 N MiB buffer，属于启发式 eviction，
不是“缓存一定冷”或“测到 HBM 带宽”的证明。压缩、cache policy 和硬件差异都可能影响它。
CSV 的 `logical_io_gbs_estimate` 是算法逻辑字节量 / 时间，不是实测 DRAM traffic。
HBM/L2 traffic 必须从 profiler 检查；不能拿 warm L2 结果除以标称 HBM 上限当成超越硬件。[CUDA-BEST]

GEMM FLOPs 使用 `2*M*N*K`。Attention 使用 causal 有效 token pair 的 FLOPs，
不是把被 mask 的整张矩阵乘法都算成有效工作。不同实现实际执行 FLOPs 可不同。

## 低精度

A4 的 `--quant-scope prepacked` 不含量化与可选 device-native repacking；
`--quant-scope end_to_end` 包含输入 B 转置、提供的 reference quantizer、repacking 和 GEMM。
提供的 quantizer 是用于 correctness 的多步 PyTorch 实现，不是优化过的生产量化 kernel。
因此它的端到端结果用来说明成本归属，不能代表 FP4 硬件的最佳系统性能。

`kernel_*` 误差对比 **解码后的量化输入**所对应的 FP32 GEMM；`quantization_*` 误差则对比原 FP16 输入。
不能用放宽 kernel tolerance 吞掉量化误差。MXFP4 编码与 scale 约定见 A4 README。[PTX]

## CUDA Graph 与 profiling

A5 的 `--graph` 捕获预热后的稳定输入/输出地址，单独记录 `cuda_graph_operator`。
缓存命中、Python launch gaps 和 graph replay 的变化要分开解释。
`peak_extra_allocated_bytes` 只统计 PyTorch allocator 可见的额外显存，不涵盖所有 driver/JIT 内部显存。
基线取在 `--evict-mb` 的 eviction buffer 分配之后，harness 自己的 buffer 不会被算成算子的 workspace。

```bash
bash scripts/profile.sh nsys a3 --op gemm --variant tma_wgmma
bash scripts/profile.sh ncu a1 --op transpose --variant padded
bash scripts/sanitize.sh a3
```

Nsight / sanitizer 采集和正式计时分开运行。采集范围默认包括 correctness/warmup，
分析时选择学生 kernel；可用 `NCU_KERNEL_REGEX` 缩小范围。metric 名称按实际 Nsight/架构选择，
本项目不写死可能不存在的 counters。每个架构点需附源码、PTX/SASS 或 trace 证据。
`profile.sh` 默认跑 smoke shape（例如 37×65 的转置），只适合确认采集流程；
要看 bank conflict、L2/DRAM traffic 这类证据，在命令末尾追加 `--suite full`。

## 指令与资源证据从哪里拿

- **A1 / A2（CUDA C++ extension）**：`MGPU_BUILD_VERBOSE=1` 会打印 nvcc 命令行和 `ptxas -v` 的结果
  （每个 kernel 的 register 数、shared memory、spill）。只有真正发生编译时才有输出；源码未变时
  PyTorch 复用缓存，缓存目录是 `TORCH_EXTENSIONS_DIR`，默认在 `~/.cache/torch_extensions/` 下。
  同一目录里的 `kernels.cuda.o` 可以用 `cuobjdump -sass` 反汇编，在其中定位 MMA / `cp.async` 对应的指令。
- **A5（Triton）**：编译产物在 `TRITON_CACHE_DIR`（默认 `~/.triton/cache`）里，含 `.ptx` 和各级 IR。
- **A3 / A4（CuTe DSL）**：按所装版本的官方文档导出 PTX / 反汇编，本项目不替你固定这条路径。[CUTLASS]

文件名和目录布局随 PyTorch / Triton 版本变化，以实际环境为准，并把所用命令写进报告。
