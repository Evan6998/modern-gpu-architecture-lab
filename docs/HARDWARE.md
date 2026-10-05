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

CPU/macOS 可用来编辑与检查 oracle，但本项目不使用 MPS 冒充 CUDA。
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

测试失败要区分：**环境不支持**、**starter 未实现**、**数值错误**、**同步/内存错误**。
这四种情况都不能用“跳过后全绿”代替正式验收。
