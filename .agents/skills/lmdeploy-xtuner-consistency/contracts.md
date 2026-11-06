# Numerical contracts

## Object of comparison

比较每个有效回复 token `y[t]` 的 `log p(y[t] | prompt, y[:t])`，明确是 [raw 还是 sampling 分布](integration.md#logprob-semantics)。第一枚回复 token 由最后一个 prompt 位置预测；使用真实 token id、position、mask，避免 shift 错一位。

固定同一权重内容与版本、tokenizer/chat template、历史、处理规则。逐 request 对齐全部位置，不能取两侧样本交集。原始 dtype 不同即违反约定，不先 cast 再声称原始字节相同。

比较原 dtype 的整数视图或原始 bytes；NaN/Inf 即使位表示一致也失败。零有效 token 是 `inconclusive`。采样 token 的 logprob 相同只覆盖这些值，不能推导 full-vocabulary logits 相同或训练质量更好。

## Per-operator contract

相同 Python 函数不保证实际 kernel、编译路径和调用参数相同。C0 每行记录实际 dispatch，C5 用同输入实验验证。

| 算子 | 需要对齐 |
|---|---|
| Embedding/encoder | 权重 bytes、gather、缩放与舍入；多模态还包括 pixel/features、占位符和位置 |
| Dense GEMM/lm_head | 输入和权重 dtype、累加精度、tile/split-K、归约树、bias/epilogue；避免按 M 切换数值路径 |
| RMSNorm/fused residual | variance dtype、eps、`1+w` 变体、block geometry、cast 位置；返回的 hidden/residual 都要保留 |
| RoPE | 表的构造设备/dtype、neox/interleaved、缩放、partial dims、q/k norm 先后、packing 位置 |
| Activation | SiLU/GELU 近似、gate/up 顺序、乘法精度和舍入次数；融合量化的边界 |
| Attention/MLA/DSA | scale、mask、GQA、sink/window、KV 读路径、split/segment、online softmax/merge 顺序、LSE 定义；MLA absorption 另核对 |
| Router/top-k | fp32 计算路径、分组、bias、normalization、稳定 tie-break、集合及顺序 |
| Experts/combine | token 排序和 padding、专家 GEMM 配置、routing weight dtype、top-k slot 归约顺序、shared expert 加法位置 |
| SSM/KDA/conv | 序列边界、chunk grid、状态 dtype、更新/读取时刻、gate 位置；prefill/decode 的递推路径 |
| Scorer | logits 交接 dtype、log-softmax kernel、温度和过滤阶段、归约顺序、目标 token gather；训练 chunk 和直接 head 路径 |

batch invariance、prefill/decode parity、跨进程稳定和跨并行布局稳定是不同性质，分别验证。kernel 能复用仍可能需要固定 dispatch 或修改 reduction；不假定 LMDeploy 默认已满足这些性质。

## Scorer dtype contract

LM head 与 log-softmax 是独立的数值边界，不能只检查最终 selected-token logprob 的 dtype。每条真实路径分别记录：

| 检查项 | 必须记录的内容 |
|---|---|
| 1. Training LM head weight dtype | XTuner forward 实际读取的 head 参数；区分存储值与 FSDP compute copy |
| 2. Rollout LM head weight dtype | LMDeploy load/post-load 后实际读取的 head 参数；不由训练配置推断 |
| 3. Hidden state dtype before head | 两侧进入 head 前的 dtype，以及 head 入口 cast 后实际送入 GEMM 的 dtype |
| 4. GEMM accumulation dtype | 实际 kernel 的乘法/累加精度、reduction、epilogue 与舍入点；输出 FP32 不证明 GEMM 全程 FP32 |
| 5. Logits output dtype | head 原始输出及其 bytes，记录任何后续 `.float()`、cast 或 buffer 写回 |
| 6. LogSoftmax computation dtype | 实际输入、内部计算/归约精度和输出 dtype；BF16 接口不代表内部累加一定为 BF16 |
| 7. Final selected-token logprob dtype | gather 后、传输/序列化前、训练 capture 处的 dtype 与 bytes；单独记录传输中的转换 |

已检查的 LMDeploy `lmdeploy/pytorch/models/utils/model.py::DeployModelMixinV1.get_logits` 先将 hidden states 转为 `lm_head.weight.dtype` 再调用 head；`FusedLogitsProcessor` 的 `raw_logprobs` 分支直接执行 `scores.log_softmax(dim=-1)`，没有显式 `.float()`。XTuner 的 `LogProbContext` 普通/chunk 路径和 `GRPOLossContext` 在 `F.linear` 后显式将 logits 转为 FP32，再调用 `gather_logprobs`。

因此，当 head 输出为 BF16 且引擎中间没有其他 upcast 时，两侧进入 log-softmax 的值域/dtype 路径分别是 BF16 与 FP32，不能直接声称原 dtype bitwise 相同。即使序列化或 capture 最后将两者都转成 FP32，也不能恢复已在 BF16 log-softmax 输出中丢失的位。`ModelAgent` 的 input scoring 与在线 sampling 路径均须追踪，不能只检查其中一个。

两个配置相互独立，默认均为 `False`：

- Training：`xtuner/v1/config/fsdp.py::FSDPConfig.fp32_lm_head`。
- Rollout：`xtuner/v1/rl/rollout/worker.py::RolloutConfig.fp32_lm_head`；LMDeploy adapter 将其传入 `hf_overrides`，再影响 head 构造。

后续 runtime 验证至少分别覆盖以下两组；每组独立保存上述七项和 C5 双路径证据：

| Training / Rollout 配置 | 检查重点 | 未经实验不能推导 |
|---|---|---|
| `False / False` | 实际 head dtype、BF16 head 输出、两侧 log-softmax 输入和输出；不要把默认配置当作已对齐 | head 均为 BF16不意味着 scorer 相同 |
| `True / True` | 两侧 head 参数和输入 cast 是否真正为 FP32，kernel/归约与 scorer 是否一致 | 两个 flag 为 True 不保证 bitwise |

若两侧配置不一致，先记录它们实际产生的数值路径，不用一个 flag 代替另一个。选择统一 FP32 或其他共同路径前，先在同一 captured logits 上比较 scorer，再比较真实 head 输出及完整路径；将选择、代价与证据冻结进配方，不为通过检查而静默更改用户精度。

本表定义待执行的验证范围；文档修改本身不表示已运行这两组 GPU 实验。

## Discrete decisions

优先比较离散状态，再分析最后一位误差：专家 id 和 slot 顺序、DSA/稀疏索引、top-p/top-k 保留集合、约束解码 mask、capacity/padding 决策。

R3 若可用，只重放已捕获的专家 id；router logits 和连续权重仍从当前参数计算并保留梯度。记录自然 routing 与 replay 的差异率。强行固定 ids/indices 是有效的定位实验，但只证明固定选择后的路径；不能作为自然 router 已对齐的结论。

## Weights and precision chain

按 tensor family 记录：

```text
A：优化器管理的参数/master
  → B：XTuner forward 实际使用的值
  → C：导出并传输的权重、scale 和布局
  → D：LMDeploy load/post-load 后 kernel 实际读取的值
```

实际测量 A/B 的 dtype，不照搬 Molt 的 fp32-master 假设。检查 XTuner FSDP、float8、保留 fp32 参数和 offload 的实现。B 与 D 在同一逻辑布局、同一解量化规则下对应，只有同 dtype 的对应值才比较原始 bytes。

映射覆盖参数和 buffer：fused QKV/gate-up、专家重排、tied head、norm、校准 scale、派生状态。每个 missing/extra item 有理由。量化及 scale 计算在完整逻辑块上定义，分片边界不能偷偷改变统计。

FP8、KV 量化、激活量化分别盘点；格式名或 `enable_float8` 相同不说明量化范围、block/tensor scale、rounding 和实际 GEMM 一致。XTuner 已有 float8 算子不等于与 LMDeploy 的 FP8 路径相同。

需要 QAT 时，训练前向使用部署量化规则的 `DQ(Q(master))` 并调用相同 kernel。写明 STE、dX 使用 master 还是 dequantized weight、激活参考值和 scale 梯度。在线量化或 QDQ 只解决部分值差，不替代算子对齐。LoRA merged/unmerged、adapter 和 draft 也逐项核对，不继承原栈的支持结论。

## Forward and backward boundary

前向调用推理 op，保存训练反向需要的输入、LSE、routing ids 和状态；反向可以复用 native 数学。必须验证参数身份、版本、cast、collective 和 normalization，禁止直接返回 detached 推理值当作可训练输出。

前向 parity、梯度合约、有效参数更新是三项证据。状态 checkpoint/recompute 必须读到同版本参数、RNG 和离散选择。注意 torch compile/fusion 可能改变 op 边界和 arithmetic；graph 与 eager 各验各的范围。

## Changing arithmetic

双方共同改变 tile、segment、量化或归约树时，保留旧 baseline，新建该数值路径的 rollout 和证据。不能宣称新路径仍保证旧输出不变；重验受影响的前向、梯度、质量和配对性能。
