---
name: lmdeploy-xtuner-consistency
description: Align XTuner v1 training with LMDeploy rollout at the per-token logprob byte level. Use for train/rollout mismatch diagnosis, batch invariance, aligned forward kernels, or consistency acceptance on this stack. Covers PyTorch and explicitly scoped TurboMind paths; distinguishes raw-logprob parity from actual sampling-policy parity.
---

# LMDeploy + XTuner train–inference consistency

让 XTuner 在相同权重、token 历史和 mask 下重算的有效回复 token logprob，与 LMDeploy 返回值逐位一致；随后验证反向、更新后同步、恢复训练、质量和性能。

这是原 [train-inference-consistency](../train-inference-consistency/SKILL.md) 的独立适配。这里提供工作方法和源码导航，不代表 LMDeploy 或 XTuner 已实现通用对齐开关，也不继承原版参考分支的模型支持或通过记录。任务只要求诊断或前向对齐时，按对应范围结束。

## Start by task

| 任务 | 阅读顺序 |
|---|---|
| 新模型的算子诊断或 aligned forward 实现 | [集成入口](integration.md#source-map) → [比较对象](contracts.md#object-of-comparison) → [scorer dtype](contracts.md#scorer-dtype-contract) → [算子盘点](training-side.md#module-inventory) → [C0–C5](workflow.md#gate-table) |
| Backward 与 proxy RL | C5 通过后，[梯度验证](training-side.md#gradient-validation) → [C6–C8](workflow.md#gate-table) 和 [两次更新](workflow.md#two-step-rl) |
| Full model | 当前候选 C0–C8 证据齐备后，[完整模型准入](workflow.md#entry-to-the-full-model) → [C9](workflow.md#gate-table) |
| 已有任务出现 mismatch | [初筛](debugging.md#triage) → [隔离两侧](debugging.md#isolate-the-side) → [保存与比较](debugging.md#save-and-compare) |
| 最终验收和提交 | [范围与配方](acceptance.md#one-recipe-two-arms) → [同步更新](integration.md#synchronous-run) → [C10](workflow.md#gate-table) → [证据](evidence.md)；完整交付覆盖 C0–C10 |

C8 通过只覆盖代理模型的更新链；full-model 验证属于 C9，最终验收属于 C10。

## Essential contracts

- 固定权重内容、token、位置、mask 和概率语义；按 request id、逻辑 token 位置比较原始 dtype 的位表示。有效 token 必须大于 0，NaN/Inf 必须为 0。
- **先区分 raw forward parity 与 sampling-policy parity。** 已检查的 XTuner LMDeploy 适配器使用 `raw_logprobs`；LMDeploy 在 temperature、penalty、mask 等处理前计算它。raw 值相等不能证明与实际采样分布相等，详见 [logprob semantics](integration.md#logprob-semantics)。
- 推理内核定义前向数值；训练侧复用实际执行的底层算子，保留 XTuner 参数、分片和优化器。反向按书面梯度约定验证，不要求逐位一致。
- BI=0 表示用户配方的原始数值路径；BI=1 表示候选对齐路径。这是实验标签，不是现成 CLI 参数。选择实际实现的配置入口并记录；不照搬 Molt/vLLM 参数。
- 从实际 worker 记录 LMDeploy、XTuner、torch、编译扩展和 GPU 架构。PyTorch、TurboMind、不同 attention/MoE 后端分别验收；同名模型不等于同一数值路径。
- 一致性比较在同步、同权重版本、更新之前进行。异步旧策略样本的差异另列，不能通过更换内核消除。
- 先在保留真实维度和全部算子类型的浅层代理模型定位，再扩大规模；固定 rollout 回放不替代真实训练入口的更新后验证。
- 未执行、零有效 token、指标缺失、只跑到诊断替代路径，都不能标记 `pass`。前向、梯度、质量和性能分别验收。
- 默认沿用原 skill 的 rollout 吞吐损失预算 20%，在 C1 与用户既有目标核对；预算调整写明依据。它是验收标准，不是性能承诺。
- 以用户已授权的范围为准；C1 基线和 C9 完整配方尚未确认时，先完成可审阅材料。任务是编写 skill 时不启动训练或占用 GPU。

## Files

| 文件 | 内容 |
|---|---|
| [integration.md](integration.md) | XTuner/LMDeploy 源码入口、配置边界、logprob 语义、权重更新、后端差异 |
| [contracts.md](contracts.md) | 相等的定义、算子数值约定、离散选择、精度与量化链 |
| [training-side.md](training-side.md) | 算子盘点、共享 forward、native backward、SP/EP/TP、梯度验证 |
| [workflow.md](workflow.md) | C0–C10、任务范围、代理模型、两次真实更新、扩大规模条件 |
| [debugging.md](debugging.md) | A/A、A/C、原始张量比较、第一处分歧、cache/graph/refit 排错 |
| [acceptance.md](acceptance.md) | 配对基线、质量、性能预算、checkpoint/resume、报告格式 |
| [evidence.md](evidence.md) | 证据标签、执行记录、双路径验证、变更后的最小重验范围 |

## Terms

| 术语 | 含义 |
|---|---|
| A/A | 同一侧在相同输入、权重和状态下重复计算 |
| A/C | 同一序列改变邻居、顺序或 packing 后重新计算 |
| BI | batch invariance；请求的数值不受 batch 邻居影响 |
| proxy | 保留真实 hidden/head/expert 几何与必要层邻接的浅层模型 |
| R3 | 重放 rollout 专家 id；不重放连续 routing weights，不自动修复其他算子 |
| refit | XTuner 更新后向 LMDeploy 部署新权重，包括 load/post-load 处理 |
| SP | XTuner 序列并行；以实际 SequenceContext、mesh 和 gather/split 语义为准 |
| closure | 结论适用的模型、权重、代码/镜像、精度、拓扑、执行模式和状态范围 |
| oracle | 不调用待测 backward 的独立前向或梯度参考 |

概念来源：[vLLM train–inference consistency 文章](https://vllm-project.github.io/2026/10/14/train-inference-consistency.html)。原文实验结果不作为本栈的证据。
