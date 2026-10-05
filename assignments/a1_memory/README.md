# A1 · SIMT 与内存层次

**传统 SIMT → Volta 同步语义 · CUDA C++**


## 任务

依次完成 `transpose` 的 naive / tiled / padded 三版，以及 `softmax` 的 shared / shuffle 两版。
保留旧版，不能让三个 variant 都指向同一个优化 kernel。
转置用于观察 coalescing、shared-memory tiling 和 bank conflict；softmax 用于练习稳定 reduction、
warp mask、跨 warp 合并、register pressure 和 occupancy。[CUDA-BEST]

| variant | 这一版必须体现的机制 |
|---|---|
| `transpose` / `naive` | 每个线程一个元素，global → global 直接搬；读写至少一侧不 coalesced，作为 baseline |
| `transpose` / `tiled` | 经 shared-memory tile 中转，使 global 读和 global 写都 coalesced |
| `transpose` / `padded` | 与 tiled 相同的算法，只改 tile 布局（例如每行多一列）来消除 bank conflict |
| `softmax` / `shared` | 每行的 max 与 sum 用 shared-memory 归约 |
| `softmax` / `shuffle` | warp 内用 `__shfl_*_sync` 归约，再做跨 warp 合并 |

## 接口与边界

```python
transpose(x, out, *, variant="naive", config=None)  # FP32 X[M,N] -> out[N,M]
softmax(x, out, *, variant="shuffle", config=None)  # FP32 X[rows,width] -> 同形 out
# config：transpose 只读 {"tile_m": 32}，softmax 只读 {"warps": 4}；其他 key 直接报错
```

`tile_m` 是 tiled / padded 的 tile 边长（naive 用作 2-D block 边长），`warps` 是 softmax 每个 block 的 warp 数。
它们原样传进 `csrc/kernels.cu` 的 dispatcher。只需支持你做消融的少数几个取值，其余用 `TORCH_CHECK` 拒绝；
不要静默换成别的值，CSV 里记的是请求值。

所有矩阵 contiguous、row-major、有限值、非空；输入不能修改，out 不得 alias 输入。
softmax 只对最后一维，必须数值稳定。支持非整齐 shape，不支持 autograd。
**实现位置：`csrc/kernels.cu`**。bindings、当前 stream 获取及 lazy compilation 已提供。
不要依赖 warp 内无条件 lockstep 来省掉必要同步。

## 公开测试

转置覆盖 1×1、非方阵、tile 边界与尾块；softmax 覆盖宽度 1/31/32/33/127/128/129/1024/8192/8193、
±10000 常数行、随机较大 logits、概率和、输入未修改。NaN out 检查会暴露没有写全的输出。

```bash
python -m pytest assignments/a1_memory -m 'not gpu'
python -m mgpu.grade a1 --variant naive
python -m mgpu.grade a1
python -m mgpu.bench a1 --op transpose --variant padded --suite full
python -m mgpu.bench a1 --op transpose --variant tiled --suite full --config '{"tile_m":16}'
python -m mgpu.bench a1 --op softmax --variant shuffle --suite full
python -m mgpu.bench a1 --op softmax --impl library --suite full
```

## 实验与提交

固定 workload 在 `benchmark.json`。只做 block/tile 布局的少量对照，不做无限搜索。
提交各版本 latency、逻辑 bytes、profiler 的实际 L2/DRAM traffic、shared-memory bank-conflict 证据。
至少解释一个“occupancy 更高但速度没有更快”的配置。不要把 cache 带宽叫作 HBM 带宽。

## 提交与评分

填写本目录 `REPORT.md` 和 `submission.json`，附源码改动、测试结果、CSV 与必要的 trace/PTX/SASS。
每题 100 分：正确性 40、架构证据 30、测量消融 20、报告 10。
完整规则见 [GRADING.md](../../docs/GRADING.md)，资料标签见 [SOURCES.md](../../docs/SOURCES.md)。
