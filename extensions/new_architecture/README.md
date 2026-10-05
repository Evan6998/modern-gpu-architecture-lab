# 新架构迁移附录 · Rubin preview

这不是第七道长作业，也不是六题完成的前置条件。
2026-10-05 查阅的 CUDA 官方首页列出 CUDA 13.4 的 **Rubin preview**。
文档中有预览支持，不代表你已经有该硬件，也不代表任意 PyTorch/CuTe build 都能用它。
来源：[CUDA-CURRENT] 与 [PTX]，见 [SOURCES.md](../../docs/SOURCES.md)。

只迁移 A4 的同一个 GEMM：确认 compiler target → 审查新 ISA/同步差异 → 原生编译 →
在真实硬件上运行既有正确性集 → 与合适 baseline 同口径比较。
不要用 `CC >= 10` 之类规则自动放行全部未来架构，也不要假设 Hopper 专有指令向后兼容。

```bash
python extensions/new_architecture/probe.py --output results/new-architecture-probe.json
# 查看 compiler 列出的目标后，再显式选择真实支持的 target：
# python extensions/new_architecture/probe.py --target sm_... --output results/new-architecture-probe.json
```

probe 只记录 nvcc 的支持目标，并可编译一个普通 elementwise kernel。
**普通 kernel 编译成功不证明使用了新 Tensor Core、collector、同步机制或硬件可用。**
需要自己补充目标 feature 的最小实验和指令证据，再决定是否启用硬件 gate。

在 `PORT.md` 中把文档确认、generic compile、feature compile、hardware correctness、hardware performance
分别标记为 verified / not tested，不得互相替代。没有 GPU 时停止于编译验证，不填性能数字。
