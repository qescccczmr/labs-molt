# XTuner aligned forward and backward

## Module inventory

C0 从 LMDeploy 实际模型文件和 backend dispatch 开始，再找 XTuner 的对应模块。每个 logprob 路径上的模块一行：

| 模块/语义边界 | LMDeploy 实际 op/源码 | XTuner 实际 op/源码 | 权重映射与 dtype | forward verdict | backward verdict | 覆盖实验 |
|---|---|---|---|---|---|---|
| 每种真实模块 | kernel、参数、编译分支 | class、调用点、head/loss | A→B→C→D | 已相等/换训练 op/需改引擎/缺接口 | reuse/glue/new | shape、路径、artifact |

清单包括 embedding、encoder、norm、residual、RoPE、attention/MLA/稀疏索引、router、experts、状态层、lm_head 和全部 scorer。冻结模块仍在数值路径上，通常仍须向上游传播输入梯度。

代理模型减少深度，保留真实 hidden/head/expert 维度、所有层类型、必要邻接/共享状态和并行整除条件。记录 feature→proxy layer→test→未覆盖项；缩小几何会改变 kernel 路径，不能代替真实 shape。

## Call kernels

XTuner 保留自己的参数容器、FSDP/mesh、优化器、模型入口和 SequenceContext。前向使用当前已 gather 的对应参数，调用 LMDeploy 的底层算子；不构造第二份训练参数，不搬入 LMDeploy scheduler、engine module 或专用 process group。

按 `LMDeploy nn → backend impl → kernel` 跟踪真实调用，而非把某个类名当作数值实现。在两侧使用相同权重和 captured 输入对单算子比较，记录实际 kernel、tile、accumulation/cast 和 compile 边界。修改共享 kernel 时两边一起部署。

特别检查 XTuner 的 head：`LogProbContext` 普通/chunk 模式和 GRPO 等 loss 会直接调用 `F.linear(hidden, head_weight, bias)`。只 patch `nn.Linear` 可能不覆盖这些调用点；C5 要分别证明它们执行了候选 head 和 scorer。

若当前 LMDeploy backend 没有可复用的稳定 op，先列接口缺口和最小改动，再实现；不得用近似 torch 重写声称已调用相同内核。TurboMind 路径必须先证明绑定可用。

## Layout, state and parallelism

- XTuner 记录 FSDP、TP/EP/SP mesh 与 SequenceContext；LMDeploy 单独记录 TP/DP/EP、attention/expert 分片和 engine 数。不要用一边的 world size 推另一边的布局。
- 将 capture 按全局 token/head/expert 坐标恢复，去 padding，避免把分片存储顺序当作逻辑顺序。SP 的 split/gather、loss mask 和 shifted_labels 都验证。
- 不同 TP/EP degree 可能有不同 reduction tree。单卡通过不覆盖多卡；固定 NCCL 算法也不自动证明不同拓扑相同。先分别证明单侧 A/A、A/C，再做跨栈对比。
- router ids 按当前格式的全局 layer、token、expert slot 映射；确认 prompt/tool tokens、缺失 routing 和 sentinel 的处理。不要照搬其他框架的 layer index 约定。
- packed attention、KDA/SSM/conv 的状态都必须遵守各序列边界；正确 attention mask 不能证明 recurrence 没有串序列。
- 覆盖不等长序列、chunk/page/tile 边界前后、空专家、近 tie、prefix hit/miss、混合 prefill/decode。recompute 保存或恢复相同的离散状态和 RNG。

## Backward contract

优先复用 XTuner/torch 或已用训练库的 same-math backward；没有整个 fused segment 的 backward 时，用已有可微 op 拼接，确有必要才新增 kernel。

| 类型 | 做法 |
|---|---|
| 只是累加/舍入不同 | 对 LMDeploy visible forward 接 native same-math backward |
| 量化/rounding | 先写 STE 和 dX 权重约定，再复用对应 GEMM/attention backward |
| routing/复合 segment | 保存离散选择，拼接 grouped GEMM、activation、combine 的梯度；连续 routing weights 保持可微 |

每类 op 写明：可训练变量、visible forward、输入/权重/累加 dtype、dX/dW 公式与参考值、scale/STE、固定离散状态、collective、loss normalization、saved/recompute 数据及参数版本。

`autograd.Function.forward` 调 LMDeploy op，`backward` 用这些输入重算 native 数学或调用训练 kernel。不得混用不一致的 attention O/LSE/P；需要时在 backward 内用同一训练 forward 重算其全部中间量。仅把输出 `detach()` 会断梯度。

## Gradient validation

前向为 0 diff 后仍单独完成：

1. **模块 VJP**：独立 oracle、不调用待测 backward；相同 dY、相同 dtype、真实 shape，比较 dX、dW、bias/router/scale 等参数梯度和一次 optimizer update。容差先固定。
2. **近似合约**：有 STE、量化或 dX 使用 master 近似时，与另一约定逐层比较 cosine、norm ratio 和偏差；无近似则写明。
3. **质量在范围内时做代理 SFT 三组**：BI=0 原路径、aligned forward+生产 backward、相同 aligned forward+独立 oracle surrogate。固定初始权重、数据/顺序、loss normalization、学习率、optimizer、clipping、recompute 和 batch；报告验证集 NLL、loss、梯度/update norm。

用已知错误检查验证器，例如将某模块梯度缩放 1.05/1.25；必须能判失败。不能看到结果后放宽阈值。检查 optimizer 实际消费的梯度和参数更新，而不仅是 backward 返回了 finite 张量。

初始 loss 不同先查 forward/data/mask；同权重 loss 相同但单步更新不同先查 backward、SP/EP 缩放、normalization、clipping 和 optimizer。后期异常保存首次异常快照，再做同权重对照，不反复跑整段曲线猜原因。

更新幅度以实际 optimizer/master 参数为准；小更新未跨越低精度舍入边界，不代表未更新。相反，只有 step 计数增长、全零 advantage 或被 IS mask 丢弃的 batch 不算有效更新。
