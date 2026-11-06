# Paired acceptance

## One recipe, two arms

BI=0 是用户原始路径，BI=1 是候选 aligned 路径。两组使用同一用户配方、镜像、checkpoint、数据/奖励、长度、采样设置、batch、optimizer、offload 和并行布局；记录对齐开关的实际入口和它隐含的 kernel/config 变化。

这两个标签不代表 XTuner 或 LMDeploy 自带 BI flag。候选实现应有清晰配置边界；在同一 fixed rollout 上证明关闭对齐时原路径 bytes 不变。进程级 patch 的两组在独立进程构建，避免两个 arm 实际都被 patch。

若候选需要 R3、graph/cache 或其他配置变化，明确它是 treatment 的一部分还是额外 confounder；能成对保持的保持一致，不能保持的定量列出。IS correction、过滤策略在两组相同，ratio reweighting 也可能改变梯度，不把全部质量变化归因于 kernel。

C1 准备用户可审阅的 baseline 与预算；C9 准备完整模型配方与代理证据。沿用已有授权，不为已确认事项重复询问。正式 workload 的变化另记，不能为使比较通过而缩短长度、改精度、关目标特性或换数据。

每组使用独立 checkpoint、输出和编译缓存路径，避免误恢复另一组。配方数据与 eval set 分开；从最终 resolved config 做 diff，不能只比较启动脚本变量。

## Quality

前向 0 diff 不证明质量、梯度或收敛。冻结 eval 数据、问题 id、解码设置、样本量、统计方法和通过线，比较逐问题结果：both_correct、only_BI0、only_BI1、both_wrong。greedy 文本相同不代替 logprob bytes 检查。

SFT 三组按 [training-side.md](training-side.md#gradient-validation) 执行；短 RL 更新验收只证明更新链。未覆盖训练质量时写 `quality not covered`，不额外启动长期实验。正式质量接受用户既定标准，不把本 skill 的默认值当成新业务要求。

## Performance budget

分别报告：固定 workload 的 rollout tokens/s、训练 forward+backward 时间、整个 RL step makespan、显存峰值。它们不能互相替代。

默认预算沿用原 skill：各正式并发档 `c` 下，

```text
deficit(c) = 1 − T_BI1(c) / T_BI0(c) ≤ 20%
```

先与用户既有目标核对并冻结；预算放宽须有明确授权和理由。超过预算时 C10 是 `fail`，不能称已交付。训练侧 overhead 无默认 20% 上限，单独测量并按约定判断。

在同节点上顺序运行两组，匹配请求/token/长度表、EOS 规则、prefill、并发、拓扑和资源占用。warmup 完成后测量；窗口里有 JIT/autotune 重编译、失败重试或资源竞争则该 pair 作废并保留原因。

先量同配置重复噪声，再交替执行 BI0/BI1 和 BI1/BI0，保留原始 pairs；逐档先计算配对 ratio，再取约定的中位数，不用不同 workload 的平均吞吐相除。多引擎以同一公共窗口的总完成 tokens/总时间计量，不相加各自不同时间窗的速率。

优化顺序按 E2E profile 决定：graph 路径与结构问题 → attention 分段 → router/expert dispatch/GEMM → dense GEMM → 其余热点。每个 arithmetic 改动重验相应 C3/C5 与梯度，再做配对 E2E；微基准最好不代表 E2E 最好。

## Checkpoint and resume

使用当前 XTuner 的真实 checkpoint 格式和保存/恢复入口，不套用 Molt 的 `--ckpt.*`。检查参数/master、optimizer/scheduler、RNG、训练 step 和模型特有状态，推理 cache 不作为跨权重有效状态保留。

恢复后确认训练状态与保存前对应，首次导出/refit 的逻辑权重/scale 正确，所有 engine 部署完成，第一轮同版本比较为 0，再检查一次有效更新。只证明 checkpoint 文件存在或 fresh process 冷加载通过不够。

记录 resume、sleep/wakeup、graph replay 和 prefix-cache 生命周期。offload、CPU/GPU optimizer 路径按配方保留，不假定不同 optimizer 实现逐位相同。诊断 eager 或 cache-off 通过只覆盖该诊断路径。

## Provenance and delivery

报告运行时实际 import 路径、代码及 dirty patch、镜像 digest、扩展/JIT hash、GPU、resolved config、tokens/weights digest。交付镜像/代码与测试对象不同则重新检查对应 closure；容器 tag 或主仓库 SHA 相同不保证 worker 加载相同代码。

改 skill 的 PR 只声明文档和接口核对，不声称完成对齐实现、GPU gates 或模型支持。实现跨库变更时按用户授权在各库分别提交最小 PR，链接依赖和验证结果；本 skill 不授权自行修改或发布其他仓库。

## Report template

```markdown
Scope: inference / fixed-forward / full-training
Probability contract: raw / sampling; exact processing and source
Candidate: model, weights/token hashes, precision, trainer/engine layouts
Runtime: LMDeploy/XTuner/torch, imported sources, image/patch/native hashes, GPU
Recipe: BI0/BI1 resolved diff, baseline/recipe confirmation, confounders

Gates C0–C10: label, reason, artifact; not-run rows explicit
C5 per path: mode, grad enabled, kernel call, valid, bitwise count, nonfinite
C8 per θ: version/hash, mask identity, compare; per update: grad/update/refit/exit
Gradient: contract, independent oracle, thresholds, negative control, coverage
Quality: dataset/settings, paired outcomes, criterion, covered range
Performance: per-c raw pairs, ratio, median deficit, budget, memory, exclusions
Resume/lifecycle: each executed path and first post-restore compare
Remaining gaps: unsupported paths and the smallest next experiment
```
