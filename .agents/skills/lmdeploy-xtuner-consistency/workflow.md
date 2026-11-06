# Workflow: C0–C10

## Task scope

先明确用户要求的是 raw parity 还是 sampling-policy parity，以及推理、固定轨迹前向、完整训练中的哪一层。已授权范围内推进；表格不自动扩展为 GPU 实验或长跑要求。

| 范围 | 执行 gates | 不得推导 |
|---|---|---|
| 推理内部一致性 | C0–C4 引擎部分、对应 C10 | trainer/backward/训练通过 |
| 固定 rollout 的训练前向 | C0–C5 | C6–C9、有效更新、质量通过 |
| 交付完整 aligned training | C0–C10；质量按约定覆盖 | 两次更新不代表收敛或长期稳定 |

## Gate table

| Gate | 工作与通过条件 | 保存的证据 |
|---|---|---|
| C0 Inventory | [每个模块](training-side.md#module-inventory)、head/scorer、权重映射、backend 可调用性都有 verdict；缺接口单列 | inventory、版本、概率语义 |
| C1 Baseline | 冻结用户配方、BI=0 实际配置、质量/梯度范围与性能预算；测量重复噪声，基线尚未授权时提交确认 | launch/config、镜像、吞吐、调参记录 |
| C2 Proxy | 浅层模型保留所有算子类型、真实几何、必要状态与并行条件 | checkpoint、coverage、未覆盖项 |
| C3 Engine | 同 token 历史下独立/合批、顺序、prefill/chunked/decode、混合步、cache 和 graph 目标路径逐位一致 | 每格 valid/mismatch/nonfinite、最小失败输入 |
| C4 Structure | profile 真实 shape 的引擎和训练前向；排除逐 token prefill launch、同步读回、重复分配/拷贝等结构问题，成本估算在预算内 | profile、热点、预算估计 |
| C5 Forward | 相同 θ 和 rollout；训练 old-logprob 与带梯度 loss 两条实际路径分别 A/A、A/C；bytes 相同且 valid>0、nonfinite=0 | 双路径 receipt、原始张量与比较报告 |
| C6 Backward | 每类 op 有 reuse/glue/new 结论和梯度合约，目标参数有 finite 梯度 | 合约、参数覆盖 |
| C7 Gradients | 独立 VJP oracle 和单次 optimizer update 在预设容差内；注入已知错误判失败；质量范围需要时做 SFT 三组 | 原始误差、阈值、负对照、曲线 |
| C8 Proxy RL | 经真实 XTuner 入口，θ0/θ1/θ2 三次比较为 0、两次有效更新、完整 refit、干净退出 | 各 θ 的 hashes、counts、更新与退出记录 |
| C9 Full model | 完整配方按用户授权确认；当前候选的代理证据齐备；先零更新 θ0，再逐轴扩大并重复 C3/C5/C8；恢复后首次比较通过 | 配方差异、准入表、resume 证据 |
| C10 Acceptance | 最终交付版本重复真实更新验证；配对质量满足约定，各正式并发下 rollout 损失≤20%或用户约定预算 | [验收报告](acceptance.md#report-template) |

C4 在引擎 C3 后、训练前向 C5 后分别检查。推理、前向、梯度、质量和性能互不替代。每个 gate 的 label 按 [evidence.md](evidence.md#labels) 记录；没有 artifact 的通过记录不能用于扩大规模。

## Cheapest useful experiment

依故障选择最小仍能复现的层级，不为每次文档或局部代码修改重跑全部 gates：

1. metadata、shape、名称/布局映射、mask/shift。
2. 单 op 同输入、重复、跨进程、实际 dtype 和 graph/compile 路径。
3. 同权重 frozen rollout 回放，原失败 A/C 加 A/A 对照。
4. 引擎调度、prefill/decode/cache、并行布局矩阵。
5. 真实反向、optimizer、参数变化、refit 后比较。
6. 完整模型、目标长度、resume、质量和性能。

缩小后故障消失不算修复；保留原复现，记录缩小遗漏的条件。三轮只得到“最终 logprob 不同”时增加探针或检查样本映射，不继续重复同一个长跑。

## Two-step RL

```text
部署 θ0 → rollout0 → trainer(θ0) 重算并比较 → backward/update → θ1
部署 θ1 → rollout1 → trainer(θ1) 重算并比较 → backward/update → θ2
部署 θ2 → rollout2 → trainer(θ2) 重算并比较 → 结束
```

每次比较都在该版本 update 前。θ2 需要第三次评分，不需要第三次更新；若当前真实入口不能无更新评分，记录并实现最小可验证路径，不把三次更新描述为两次。

每阶段记录权重 hash/版本、token hash、mask、有效 token、实际 forward、非有限数、bitwise count；每次更新记录 finite loss、非退化 advantage、梯度覆盖、optimizer step、实际参数 update norm 和 refit 完成。

正式验收使用用户真实 recipe、数据/奖励、样本数、长度、batch、precision、拓扑和学习率 schedule，只按约定缩短更新次数。微型回放不能替代正式入口。完成既定比较与性能验收后结束，不自行追加长跑并宣称长期稳定。

## Entry to the full model

扩大规模前，检查当前 candidate 的 C2 覆盖、C3、C4、C5 双路径、C6–C7 合约、C8 更新链，以及 C1 baseline 和 C9 recipe 的确认状态。代理未覆盖项仍标明；先做完整模型 θ0 零更新检查，失败即保存并缩回可复现范围。

需要真实深度、跨层状态或多机拓扑才能复现时，可按已授权范围先做最小完整模型定位；记录 proxy 不足、待区分假设、最小规模与停止条件。一次 full-model baseline 不自动证明 full-model aligned 路径。
