# A4 · Blackwell 计算模型与低精度

**数据中心 Blackwell SM100 · CuTe DSL 或 CUDA C++**


## 任务

同一个 GEMM，真正迁移到 **SM100 的 tcgen05 + TMEM**；先 1-SM，再 2-SM MMA，
最后加入 persistent tile scheduler。不能仅将 Hopper 代码重新编译。[TCGEN05][BLACKWELL]
再实现 FP8 和 MXFP4 两个低精度路径，区分 kernel 计算误差与量化误差。

## 接口与边界

```python
gemm(a, b, out, *, variant="tcgen05_1sm", config=None)
# variants: tcgen05_1sm / tcgen05_2sm / persistent
prepare_quantized(a_quantized, bt_quantized) -> PreparedQuantizedGemm
quant_gemm(prepared, out, *, config=None)
```

Dense 输入/输出与 A3 一致，M/N 为 128 倍数，K 为 32 倍数。
量化 operand 均以 **K-major 逻辑矩阵**定义：A 是 `[M,K]`，另一份是 **Bᵀ `[N,K]`**。
`mgpu/quantization.py` 提供可运行的编码、解码和数值 oracle，不要求你重写 quantizer。

| 格式 | data | scales | 数值含义 |
|---|---|---|---|
| `fp8` | E4M3FN `[rows,K]` | FP32 `[1]` | 本题采用 per-tensor scale，epilogue 应用乘积 |
| `mxfp4` | uint8 `[rows,K/2]` | uint8 `[rows,K/32]` | E2M1 + 每 32 个 K 元素一个 E8M0 scale |

MXFP4 低位 nibble 对应偶数 K，高位对应奇数 K；code bit3 是 sign，bit0–2 对应
`[0,.5,1,1.5,2,3,4,6]`。E8M0 byte `b` 的 scale 为 `2**(b-127)`，本题不生成 NaN scale。
量化采用 nearest ties-to-even；选择能覆盖 block 最大绝对值的 power-of-two scale。
这是 **MXFP4，不是 NVFP4**。硬件所需的 swizzled/scale layout 可在 `prepare_quantized` 中生成。[PTX]

**实现位置：`kernels.py` 与必要时 `student.prepare_quantized`**。
FP16 解码再调用普通 GEMM 可以通过数值检查，但不满足低精度硬件要求。

## 公开测试

Dense 三种 variant、多种 shape、重复运行；量化覆盖 zero、mixed scales、独立 codebook/nibble packing、
ties-to-even、decode oracle 和输入不变。自动 gate 只接收 SM100，不把 RTX SM120 当作相同 ISA。

```bash
python -m mgpu.grade a4
python -m mgpu.bench a4 --op gemm --variant tcgen05_2sm --suite full
python -m mgpu.bench a4 --op quant_gemm --variant mxfp4 --quant-scope prepacked --suite full
python -m mgpu.bench a4 --op quant_gemm --variant mxfp4 --quant-scope end_to_end --suite full
```

## 实验与提交

1-SM/2-SM、grid/persistent、FP16/FP8/MXFP4 各做一组对照。
提交 TMEM 生命周期、tcgen05 指令证据、低精度 scale 布局，以及两种时间口径。
提供的 reference quantizer 故意不追求性能；不要把其开销解读成硬件量化的最佳开销。
最新预览架构另见 [迁移附录](../../extensions/new_architecture/README.md)，不额外增加主线作业。

## 提交与评分

填写本目录 `REPORT.md` 和 `submission.json`，附源码改动、测试结果、CSV 与必要的 trace/PTX/SASS。
每题 100 分：正确性 40、架构证据 30、测量消融 20、报告 10。
完整规则见 [GRADING.md](../../docs/GRADING.md)，资料标签见 [SOURCES.md](../../docs/SOURCES.md)。
