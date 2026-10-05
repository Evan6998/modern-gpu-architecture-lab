# 评分与验收

每题 100 分：正确性 40、架构使用证据 30、测量与消融 20、一页报告 10。
这是一套公开自学/课程作业框架，没有所谓已配置好的 hidden grader 或排行榜。

## 自动部分

```bash
make cpu                         # harness/oracle，不计作业分
python -m mgpu.grade a1           # 正式单卡正确性 gate
python -m mgpu.grade a2 --variant async
python -m mgpu.grade a6 --nproc 2  # 分别验收 serial 和 overlap
```

单卡 grading 用 `--require-gpu --strict-hardware`，无 CUDA、错误架构、零测试、skip 都不能通过。
核心 TODO 会抛异常 / 留下 NaN；不会被测试改成 xfail，也不会自动 fallback 到参考实现。
`summary.json` 只声明自动正确性结果，`numeric_grade` 留空，不能由测试数量机械推出总成绩。

40 分正确性可按公开 cases 的通过比例审核，但重大数据损坏、遗漏 token、输入被修改、死锁或
系统性数值错误会阻止题目验收；性能无论多好都不能抵消。部分版本完成可用 `--variant` 逐项检查。

## 人工部分

30 分架构：题目要求的机制必须真实进入执行路径。源码里的注释或 variant 名字不是证据。
例如 A4 先把 FP4 解码成 FP16 再调用 GEMM，数学结果可能正确，但不满足 FP4 Tensor Core 要求。
A6 gather 所有权重到每个 rank 不满足 expert-parallel dispatch，尽管 reference baseline 正是这样做的。

20 分测量：有 baseline、同口径配置、重复样本和有控制变量的消融。无需承诺超过 cuBLAS；
优化无收益也可以得到测量分，但必须解释证据和瓶颈。不得伪造带宽、GPU 型号或跨机器比较口径。

10 分报告：按 `REPORT.md` 用一页讲清至少一个成功优化和一个失败配置，引用实际 run ID。
单独的新架构迁移附录不改变六题总分，也不得把无硬件的编译检查计作性能实验。

## 允许与禁止

可以使用框架做张量分配、测试、计时；可复用 CuTe/CUTLASS 的 layout、descriptor、同步原语，
也可引用官方例子来理解 API，但要注明来源，核心 mainloop / 调度必须满足题目要求。
A5 必须自行实现 Triton attention；A6 通信可以调用 NCCL / torch.distributed，计算复用前题。
库函数可以当 baseline，不能伪装成学生 kernel。测试不是安全隔离的对抗平台：禁止按测试 seed/shape
硬编码答案、读取 reference 输出、缓存上一轮结果或修改测试来取得通过。
