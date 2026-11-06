# LMDeploy + XTuner integration

本文件按 XTuner v1 和 LMDeploy PyTorch 源码组织。路径相对于各自仓库根目录；它们是定位入口，不是已发布 API 保证或数值通过记录。先在用户选择的分支重新检查，再写候选实现。不要把其他 checkout 的修改当成当前 worker 已加载的能力。

## Source map

| 环节 | XTuner 入口 | LMDeploy 入口或检查点 |
|---|---|---|
| 选择 rollout backend | `xtuner/v1/rl/rollout/__init__.py`，`XTUNER_USE_LMDEPLOY=1` | 确认实际启动的是 `pytorch` 还是 `turbomind` |
| 请求和服务配置 | `xtuner/v1/rl/rollout/lmdeploy.py::LMDeployWorker`，`_get_request_payload`、`_transform_rollout_config_to_server_configs` | `lmdeploy/messages.py` 中 `PytorchEngineConfig`、`GenerationConfig` |
| 配置 schema | `xtuner/v1/rl/rollout/worker.py::RolloutConfig` | 在服务进程读取最终 engine config；不是每个字段都直接透传 |
| 推理模型和算子 dispatch | 核对目标 XTuner model/module/ops 的实现 | `lmdeploy/pytorch/models/`、`nn/`、`backends/`、`kernels/` |
| rollout 采样与 logprob | 检查 token-id 请求、响应解析和 SampleParams 转换 | `lmdeploy/pytorch/engine/logits_process.py::FusedLogitsProcessor` |
| 固定 token 评分 | 按当前服务是否暴露输入评分接口核对 | `GenerationConfig.logprob_start_len`；`lmdeploy/pytorch/engine/model_agent/agent.py::_get_outputs_with_logprobs` |
| old-logprob 前向 | `xtuner/v1/rl/trainer/actor_worker.py::ActorWorker._collect_old_policy_snapshots` → `TrainEngine.forward_only` | 对比同权重 rollout 张量 |
| old-logprob head/scorer | `xtuner/v1/loss/rl_loss.py::LogProbContext` 的普通和 chunk 路径 | 与 engine head、log-softmax 的实际输入和 dtype 比较 |
| 带梯度训练 head/scorer | `ActorWorker._fit_policy_then_sft`、`_build_step_input` → 继承的 `RoleWorker._train_update` → 当前 loss，如 `xtuner/v1/rl/loss/grpo_loss.py` | 注意 loss 内部直接 `F.linear`，仅替换模型 `nn.Linear` 可能漏掉 lm_head |
| 训练 log-softmax | `xtuner/v1/rl/utils/misc.py::gather_logprobs` | engine scorer 的归约和 cast 位置 |
| 现有误差指标 | `xtuner/v1/rl/rollout_is.py::compute_mismatch_metrics` → `ActorWorker._collect_old_policy_snapshots` → `xtuner/v1/train/rl_trainer.py` | 核对真实日志 key 和覆盖的路径；目前不能假定已有 bitwise 指标 |
| 权重导出 | `xtuner/v1/rl/weight_update/weight_iterator.py::WeightIterator` | load 后的参数、scale、布局与派生 buffer |
| 权重传输 | `xtuner/v1/rl/weight_update/update_weighter.py::WeightUpdater`、同目录 `transport.py` | `lmdeploy/pytorch/engine/engine.py`、同目录 `model_agent/agent.py` 的更新和 post-load 生命周期 |
| 真实训练循环 | `xtuner/v1/train/rl_trainer.py`，用户现有 recipe | 同步、版本、暂停/恢复 rollout 的实际时序 |

在 trainer、rollout Ray actor 和实际 engine worker 内分别记录：模块 `__file__`、版本、源码 SHA/dirty diff、扩展/JIT hash、torch/CUDA、GPU sm、resolved config、环境覆盖。驱动进程的 import 路径不代表 worker 的路径。

官方入口：[LMDeploy pipeline API](https://lmdeploy.readthedocs.io/en/stable/api/pipeline.html)、[XTuner GRPO Python 配方](https://xtuner.readthedocs.io/en/latest/rl/tutorial/rl_grpo_trainer.html)。具体行为以执行版本源码和 receipt 为准。

## Training call paths

当前调用链已对照 [ActorWorker](https://github.com/InternLM/xtuner/blob/c93eb15dc0b1e4fd371b33d79bec9e1321fe20f2/xtuner/v1/rl/trainer/actor_worker.py)、[TrainingWorker](https://github.com/InternLM/xtuner/blob/c93eb15dc0b1e4fd371b33d79bec9e1321fe20f2/xtuner/v1/rl/trainer/worker.py) 和 [RoleWorker](https://github.com/InternLM/xtuner/blob/c93eb15dc0b1e4fd371b33d79bec9e1321fe20f2/xtuner/v1/rl/trainer/role_worker.py) 的源码快照；使用其他 revision 时重新核对实际实例和委托关系。

```text
old-logprob：
TrainingWorker.actor_fit()                         # trainer/worker.py
  → ActorWorker.fit()                              # trainer/actor_worker.py
  → ActorWorker._collect_old_policy_snapshots()
  → TrainEngine.forward_only(seq_ctx, loss_ctx=read_ctx)["loss"]
  → LogProbContext 普通/chunk 路径
  → F.linear(...).float() → gather_logprobs

带梯度的 policy update：
TrainingWorker.actor_fit() → ActorWorker.fit()
  → ActorWorker._fit_policy_then_sft()
  → ActorWorker._build_step_input()                 # 组装 loss context 与 items
  → RoleWorker._train_update(items)                 # ActorWorker 继承的方法
  → TrainEngine.train_step()
  → GRPOLossContext（按实际配方的 loss 类型核对）
  → F.linear(...) → .float() → gather_logprobs → policy loss
```

old-logprob 的 capture 放在 `_collect_old_policy_snapshots` 的真实 `forward_only` 输出，训练路径的 capture 放在 loss 使用的 logprob 上。snapshot 被保存或绑定到训练输入不代表训练 scorer 已被验证；C5 分开检查两条路径。

## Config boundaries

`XTUNER_USE_LMDEPLOY=1` 选择可用的 rollout 实现，不启用数值对齐。检查其他 backend 选择变量和实际实例类型，避免多后端配置歧义。

已检查的适配器构造 `PytorchEngineConfig`，并设置 `logprobs_mode="raw_logprobs"`。`RolloutConfig` 中的 TP/EP、prefix caching、prefill 上限、float8 等字段经过转换；`extra_rollout_config` 的 `lmdeploy_` 前缀也不是任意 engine 字段的通用透传接口。先跟踪字段最终传给 `serve` 还是 engine config，再给出修改。

`eager_mode`、`enable_prefix_caching`、`max_prefill_token_num`、`block_size`、`quant_policy`、`model_format` 是候选检查字段，是否可由当前 XTuner 配方设置需要逐项核对。不得复制 `--vllm.*`、`VLLM_BATCH_INVARIANT`、Molt 的 `--train.batch_invariant` 或编造 LMDeploy 等价开关。

若已有局部 BI 实现，记录开关到分支、实例、执行 kernel 和结果的调用链；仅搜到变量名不够。若没有，C0 的结果是待实现算子清单。

## Logprob semantics

必须先选择并记录比较目标：

| 目标 | 比较值 | 能证明什么 |
|---|---|---|
| raw forward parity | 原始模型 logits 的 log-softmax 中该 token 的值 | 两栈对该历史的原始模型评分相同 |
| sampling-policy parity | 真实采样处理后的分布对该 token 的 logprob | 训练使用的概率与采样行为策略一致；还需同步和更新条件 |

在已检查的 `FusedLogitsProcessor.__call__` 中，`raw_logprobs` 在 temperature、repetition penalty、guided/bad/stop mask 等处理前计算；后续还有采样过滤。XTuner 的 `gather_logprobs` 对 logits 做 `F.log_softmax`，已检查的 GRPO head 不在此处除以 rollout temperature。这只是 raw 概率语义的候选配对：LMDeploy 直接对 `scores` 做 log-softmax，XTuner 则在 head 后 `.float()`；还须满足 [scorer dtype contract](contracts.md#scorer-dtype-contract)，不能直接声称 raw bytes 相同或 true on-policy。

若用户要求 sampling-policy parity：保存实际 temperature、top-p/top-k/min-p、penalty、min-new-token/EOS mask、约束解码和归一化规则，并比较处理后的同一个分布。即使 temperature=1，其他过滤仍可能改变概率。关闭这些处理只适用于用户同意的诊断或冻结配方，不能偷偷改变正式任务。

如果当前 engine 不返回所需分布的 logprob，将该能力标记 `missing`，实现对应 capture/scorer 后再验收。不要用 raw 值替代，也不要从不同处理阶段的值倒推后宣称 bitwise。

输入评分接口在较新 checkout 中可见 `logprob_start_len`，须核对当前分支、API 暴露、`max_new_tokens=0` 的支持及 chunk 边界。固定历史评分通过不等于在线 decode 通过；cache 命中时还要证明每个被比较位置都有真实分数。

## Metrics and capture

XTuner 现有 mismatch 指标包含 KL、k3、PPL 和 logprob 绝对差，主要连接 old-logprob/IS 计算。不要假定这些指标验证了带梯度的 loss 前向，也不要假定日志已有 Molt 的 `mismatch/bitwise_frac`。

适配时在两条训练路径分别 capture 与 rollout 配对的原始张量；如需要在线统计，再新增并注明以下指标的 producer：

- `valid_tokens`、`bitwise_count`、`nonfinite`，跨分片按有效逻辑 token 求和。
- `bitwise_frac = bitwise_count / valid_tokens`，只在分母大于 0 时定义。
- `abs_diff_mean`；最大差和第一失败位置保留在回放报告中。

这些是本 skill 要求的证据字段，尚不代表运行库已有同名功能。日志前缀按当前 logger 的包装核对，不靠记忆拼 key。

## Synchronous run

从用户现有 recipe 修改，不提供虚构的一键启动命令：

1. 检查 `BaseRLTrainerConfig` 的 replay buffer、`sync_weights_interval`、`train_batch_size` 及真实消费/更新循环。`SyncReplayBufferConfig` 和 `sync_weights_interval=1` 是检查入口，不是完整同步证明。
2. 生成前确认全部 engine 已部署 θ；训练消费该 θ 的完整轨迹，禁止其他版本片段混入。记录生成、消费、更新和部署的版本/hash。
3. 在 optimizer 更新前比较；一批轨迹只做一次有效更新。若多 minibatch 更新，只有第一次更新前仍与采样权重相同，后续不能按零差验收。
4. 暂停/排空在途 rollout，完成所有 rank 的更新、load/post-load 和 cache 失效，再恢复下一轮。具体顺序按当前生命周期实现证明。

异步模式单列 policy lag，raw/processed 语义不明或权重不同的比较不能用于内核验收。

## Refit and lifecycle

核对 `WeightIterator` 的名称映射、fused QKV/gate-up 顺序、专家编号、tied weights、dtype、FP8 scale 和派生参数。追踪当前选择的 IPC/NCCL 通道以及 LMDeploy adapter 的更新边界；其他 backend 的 checkpoint-engine 支持不能直接算作 LMDeploy 支持。

HTTP/RPC 成功只证明请求完成；比较 trainer-forward 权重与 engine `load_weights`、post-load 后的实际内容。预打包权重重建、量化重排、图捕获地址和 cache 失效分别检查。每个 refit 都要等全部 bucket 和 rank 完成后才能继续生成。

冷加载、首次 refit、后续 refit、sleep/wakeup、cache hit/miss、graph replay、checkpoint resume 分开记录。新进程通过不能证明旧进程中的状态已修复。MTP/投机 draft 的权重、accept/reject 和返回 logprob 也须独立验证。

## PyTorch and TurboMind

PyTorch 路径从 `nn` 找接口，再追到实际 `backends`/`kernels`；复用底层 op，避免带入 engine process group、KV 调度和图管理。attention backend 名称不继承 vLLM 的 `TRITON_ATTN`。

TurboMind 主要算子在 `src/turbomind/` 和对应绑定中，不假定可从 PyTorch 直接调用。先判定可复用绑定、是否需新增 forward wrapper 和 backward 合约。PyTorch 代理模型的通过记录不覆盖 TurboMind。用户选择 TurboMind 时，保留该目标，清楚报告缺失接口与最小实现范围，不静默切换后端。
