# 环境与硬件

## 支持边界

| 作业 | GPU 目标 | 默认 gate | 说明 |
|---|---|---|---|
| A1 | CUDA GPU | CC ≥ 7.0 | 旧编程模型在新卡上练；实际可用 SM 还受 PyTorch/toolkit 约束 |
| A2 | 首选 A100；也可在新卡练旧模型 | major 8 / 9 / 10 / 12 | 需要 MMA、ldmatrix、cp.async；不把新卡结果标成 A100 结果 |
| A3 | H100/H200 | SM90 | WGMMA 路径限定 Hopper；不是 `CC >= 9.0` |
| A4 | 数据中心 Blackwell | SM100 | 目标是 tcgen05/TMEM；RTX SM120 不等于 SM100；其他 Blackwell target 先显式移植再加 gate |
| A5 | 首选 Ampere 或之后支持的设备 | major 8 / 9 / 10 / 12 | 使用与设备匹配的 Triton |
| A6 | 2 / 4 / 8 个 CUDA ranks | NCCL | 固定 8 experts；最多 8 ranks，避免无声明改变题目 |

这些是本项目的测试支持范围，不是所有 NVIDIA 产品的完整兼容表。
架构差异依据官方 tuning guide、tcgen05 guide 和 PTX ISA；见 [SOURCES.md](SOURCES.md)。

## 安装

CPU/macOS 可用来编辑与检查 oracle，但本项目不使用 MPS 冒充 CUDA；具体边界见本页末尾。
GPU 实验建议 Linux。先选与你的驱动/GPU 匹配的 PyTorch CUDA build，再装项目：

```bash
python -m pip install -e '.[dev]'
# A5 才需要；让安装结果与当前 PyTorch/Triton 依赖兼容
python -m pip install -e '.[triton]'
# A3/A4 选择 CuTe DSL 时才需要
python -m pip install -e '.[cute]'
python -m pip check
python -m mgpu.doctor --output results/environment.json
```

依赖下界只让 Python harness 可安装，不保证任意组合支持 SM100。
这里没有提供“已在所有 GPU 上验证过”的统一锁文件。成功搭建 GPU 环境后执行：

```bash
bash scripts/lock_environment.sh results/my-gpu-environment
```

A1/A2 的 PyTorch extension 还需要本机 C++ compiler、`nvcc`、CUDA headers 和 Ninja。
只有 PyTorch runtime wheel 而没有完整 toolkit，不能编译 `.cu` 文件。[TORCH-EXT]
C++ 标准由 PyTorch 自己决定，`mgpu/extension.py` 不传 `-std`：PyTorch 2.11 及更早的 extension
构建用 C++17，2.12 起改为 C++20，2.14.1 的头文件在 C++17 下直接 `#error`。host compiler 和 `nvcc`
要支持所装 PyTorch 需要的标准。`MGPU_BUILD_VERBOSE=1` 打印完整编译输出。

`nvidia-cutlass-dsl` 默认依赖 CUDA 12 的运行库；CUDA 13 环境装 `nvidia-cutlass-dsl[cu13]`
（2026-10-05 查阅 PyPI 4.8.0 的依赖声明），与 PyTorch 的 CUDA 大版本保持一致。

```bash
nvcc --version
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())'
```

A1/A2 默认让 PyTorch 为可见设备选择编译目标；可用 `TORCH_CUDA_ARCH_LIST` 明确设定。
针对 Hopper/Blackwell 的 architecture-specific ISA 时，遵循当前工具链关于 `sm_90a` / `sm_100a`
等目标的规则；不能拿普通 PTX forward compatibility 推断带 `a` 指令的兼容性。
A3/A4 的 CuTe backend 由学生接入，不默认帮你编译或下载 CUTLASS 源码。[PTX][CUTLASS]

## 检查 GPU 而不是猜 GPU

`doctor` 记录型号、SM、显存、SM 数量、PyTorch/CUDA、driver 输出、拓扑以及相关环境变量。
它不会安装驱动、修改时钟、修改 persistence mode 或访问远程集群。
A4 的 RTX / 数据中心区分与未来架构兼容策略在 `mgpu/hardware.py` 中明确编码。

第一次拿到一台 GPU 机器时，先跑一遍首检：

```bash
bash scripts/gpu_smoke.sh
```

它依次检查驱动与工具链、`make doctor` / `make check` / `make cpu`、A1 与 A2 的 starter 能否编译并停在各自的 `TODO`、
library baseline 能否通过 benchmark 的正确性检查，以及 `ncu` 能否读到 GPU 性能计数器。每一项都会执行，
最后汇总没通过的项。starter 编译不过是环境问题，不是作业没做。

有些 PyTorch 不像 pip wheel 那样自带 pybind11 头文件（例如以系统包形式安装的发行版），A1 / A2 的 extension
会报 `pybind11/pybind11.h: No such file`。装上 `pybind11` 模块即可，`mgpu/extension.py` 会从它取 include 路径。
`nvcc`、`ncu` 等工具不在 PATH 上时，把 CUDA toolkit 的 `bin` 目录加进 PATH。

测试失败要区分：**环境不支持**、**starter 未实现**、**数值错误**、**同步/内存错误**。
这四种情况都不能用“跳过后全绿”代替正式验收。

## 在 macOS / 没有 NVIDIA GPU 的机器上

能做（2026-10-05 在 macOS arm64、PyTorch 2.14.1 上实测）：

- `make check`、`make cpu`、`make dist-smoke`，以及
  `python -m mgpu.bench <题> --impl reference --device cpu --suite smoke`：
  检查 oracle、数据生成、计时与 CSV 管道。
- 读题、读 reference、写报告，分析从 GPU 机器拷回来的 CSV / trace。
- 纯 host 侧逻辑的原型：A4 `prepare_quantized` 的布局变换（纯 tensor 操作，可对照
  `dequantize_reference`）；A5 online softmax 的 tile 循环（先用 PyTorch 在 CPU 上对照
  `reference.attention`）；A6 的分桶、permute、split-size 与还原（单进程模拟多个 rank）。

不能做：

- 编译 `.cu`：CUDA toolkit 没有 macOS 版本，A1 / A2 的 extension 无法构建。
- 安装 Triton 与 CuTe DSL：两者都没有 macOS 发行版，A3 / A4 / A5 的 kernel 无法运行。
- NCCL：A6 的学生实现要求 CUDA + NCCL；Gloo 只用来验证 reference harness。
- 任何不带 `--cpu` 的 `mgpu.grade`，以及任何性能数字。MPS 不是 CUDA。

所以 kernel 的“改代码 → 编译 → 测试 → profile”循环应当直接放在 GPU 机器上（SSH 或远程 IDE），
macOS 只承担上面“能做”的部分。注意 A3 只在 SM90、A4 只在 SM100 上验收，开工前先用 `make doctor`
确认目标机器的 `lab_families`。

`make dist-smoke` 把 rendezvous 固定在回环地址：torchrun 默认对外通告主机 FQDN，
主机名解析不了时（笔记本上很常见）会一直重试到超时。README 里其他单机 `--standalone`
命令遇到同样的症状时，加上 `--local_addr=127.0.0.1`。
