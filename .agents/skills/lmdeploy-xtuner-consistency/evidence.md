# Evidence and experiment records

## Labels

| 标签 | 条件 |
|---|---|
| `pass` | 当前声明范围内的检查全部执行，满足预先固定的条件且有 artifact |
| `fail` | 实际数值比较或约定的质量/性能条件失败 |
| `inconclusive` | invalid：目标路径未执行；vacuous：零有效 token/零调用；fault：OOM、NCCL、actor 失败等未回答问题 |
| `not run` | 范围外或未执行；写明原因，不能升级为 pass |

非有限有效值失败，即使两侧 NaN bytes 相同。没有 bitwise producer 的日志空项不是零；正常退出与完成数值验收是独立事实。

## Dual-path receipts

C5 对同一 θ、tokens 和 mask，分别捕获：

- **路径 a：old-logprob/forward-only**，XTuner `TrainingWorker.compute_actor_logprobs` → `LogProbContext`；记录实际 model/grad mode，不为了测试强行开启 autograd。
- **路径 b：loss 使用的 grad-enabled forward**，在当前 RL loss 真正使用的 logprob 处 capture；覆盖 direct head、chunk 模式、packing、SP 和 activation recompute。

每条路径记录：实例 class 与源码、grad/mode、kernel 命中、dtype、概率语义、完整 token 集合和 valid/mismatch/nonfinite。某配置跳过 a 时标 `not run`，可按授权在 proxy 上单独运行；a 的通过不替代 b。b 的 forward 通过不替代 C6–C7。

## Candidate ledger

每个结果绑定一个执行范围：

| 维度 | 必须记录 |
|---|---|
| 模型/数据 | geometry、层邻接、checkpoint/权重内容、tokenizer、输入/token/mask hash、概率处理 |
| 代码/运行时 | 两库实际 import、版本/dirty patch、image、扩展/JIT hash、GPU sm |
| 路径 | trainer/engine backend、head/scorer、precision、quant、graph/compile/cache、cold/refit 状态 |
| 布局/workload | FSDP/TP/EP/SP 与 engine TP/DP/EP，packing、长度、并发、真实独立样本/token 数 |
| 结果 | gate、预设 criterion、counts、首个分歧、gradient/update/refit/exit、label 和 artifact |

manifest 版本、load 后权重内容、rollout artifact 分开；相同版本字符串不是相同权重。重复四个样本至 batch 32 仍是四个独立样本。

旧记录保留但不复制为新 candidate 的通过。后端、镜像、kernel、tile、量化、几何、布局或 cache/graph 改变时，先判断依赖，再重验受影响项：

| 改动 | 最小重验 |
|---|---|
| op/tile/segment/quant/scorer | 同输入 op、相关梯度、proxy C5 A/A+A/C、相应 C3 边界 |
| head/loss/compile/autograd wrapper | 两条实际路径、调用 receipt、VJP、proxy、真实 θ0 |
| packing、SP/TP/EP、graph/cache | 对应矩阵、逻辑坐标/分母、C5，再 C8 |
| optimizer、weight export/refit | 同梯度 update、A→B→C→D 内容、更新后比较 |
| 模型深度/几何/状态路径 | proxy coverage、C3/C5，再 C8 与新规模 θ0 |
| image/native build/库升级 | worker 内重新确认执行范围，复验受影响 gates；旧镜像结果不迁移 |

已证明的非依赖可记录理由；文件名相同或“只是 cleanup”不构成证明。

## False greens

- 只有 KL/PPL/绝对均值为零，或打印值相同；缺少原 dtype bit compare。
- raw logprob 相等，却把采样处理后的概率一致性标 pass。
- 两组都被进程级 patch；目标 kernel 没执行；forward 被跳过。
- 零有效 tokens、全零 advantage、IS mask 丢弃全 batch、optimizer 没改变任何训练参数。
- 代理/短序列通过被当作完整模型/长序列通过，PyTorch 通过被当作 TurboMind 通过。
- eager、替代 attention、forced ids、cache-off 的诊断结果被当作正式路径修复。
- 保存的旧 logprob 与其 detached copy 比较，绕过真实训练 loss 的 forward。
- 固定回放通过被当作真实 rollout→update→refit 链通过。
- launcher 退出成功但 recipe、某 rank 或最后阶段失败；缺失数据被当作通过样本删除。

先固定 criterion，再注入已知错误证明比较器能失败；位翻转、1 ULP、梯度 scale、错位 mask/权重版本属于不同负对照。不能看到结果后移动阈值、改输入、删除失败轮或重新抽样。

证据只回答已执行的问题。skill 文档校验、CPU 比较器校验、单 kernel、代理模型、完整模型、长期质量分别陈述，不互相替代。
