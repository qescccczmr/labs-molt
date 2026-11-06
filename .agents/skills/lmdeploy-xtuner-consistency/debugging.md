# Debugging a mismatch

## Triage

先检查以下事实，再动精度或 kernel：

1. 当前候选是否真的实现了 aligned 路径，所有 worker 是否执行它？原始 LMDeploy+XTuner 的非零差异不是“开关失效”的证据。
2. 比较的是 raw 还是实际 sampling 分布？temperature/penalty/filter 的阶段是否对应？
3. 权重内容、版本、token id、position、shift、EOS/tool/padding mask 是否一致？是否已发生 optimizer update 或部分 rollout 跨了版本？
4. 记录是否来自真实张量、有效 token 大于 0、两侧集合相同且无 NaN/Inf？现有 KL/PPL 指标并不等于 bitwise 比较。

## Isolate the side

| 控制实验 | 固定输入后改变什么 | 定位 |
|---|---|---|
| Engine A/A | 重复调用；必要时跨进程 | 单侧运行/编译/状态不稳定 |
| Engine A/C | 独立/合批、邻居、slot、顺序 | batch、tile、dispatch 依赖 |
| Engine phases | full/chunked prefill、逐 token decode、混合步 | attention split、KV/状态生命周期 |
| Trainer A/A、A/C | 原 packing 重复，再换顺序/邻居/packing | 训练前向、SP、状态边界 |
| Same logits → scorers | 同一 logits tensor 分别调用两侧 scorer | dtype、log-softmax、处理语义差异 |

两侧各自稳定仍不代表互相相同；先固定历史，停止用重新采样复现文本。相同 seed 不保证相同调度或 RNG 消耗。backend/config 变化一次只改一个因素。

## Save and compare

每个实验保留：对应 θ 的更新前 checkpoint、权重 hash、原始 token/position、request id、prompt/response 边界、mask、原始 dtype logprob、probability kind、处理参数、routing/稀疏索引和状态。生成侧与训练侧均 `clone` 后导出，避免 workspace/graph 输出被后续调用覆盖。

优先使用 tensor 文件保存；文本六位小数日志不是字节证据。若当前 XTuner 没有对应 dump/replay 接口，先加最小 capture，不套用 Molt 的参数。固定回放期间证明没有 sampling、optimizer 或权重变化；普通训练式 replay 仍可能更新，不能假定它是 frozen scorer。

按 request id 和逻辑位置对齐，集合、tokens、positions、mask、weights、概率语义必须相同，不能取交集或重新 tokenize。然后对每个序列的张量执行原 dtype 比较，例如：

```python
import torch


def compare_tokens(left, right, mask):
    assert left.dtype == right.dtype and left.is_floating_point()
    assert left.shape == right.shape == mask.shape
    assert mask.dtype == torch.bool
    x = left[mask].contiguous()
    y = right[mask].contiguous()
    n = x.numel()
    if n == 0:
        return {"label": "inconclusive", "valid_tokens": 0}
    xb = x.view(torch.uint8).reshape(n, x.element_size())
    yb = y.view(torch.uint8).reshape(n, y.element_size())
    bad = (xb != yb).any(dim=-1)
    nonfinite = (~torch.isfinite(x) | ~torch.isfinite(y)).sum().item()
    count = bad.sum().item()
    first = bad.nonzero().flatten()
    return {
        "label": "pass" if count == 0 and nonfinite == 0 else "fail",
        "valid_tokens": n,
        "bitwise_count": count,
        "bitwise_frac": count / n,
        "nonfinite": nonfinite,
        "first_valid_index": first[0].item() if count else None,
        "abs_diff_max": (x.double() - y.double()).abs().max().item() if nonfinite == 0 else None,
    }
```

调用者把 `first_valid_index` 映射回 request/绝对位置，非有限数另定位。跨 microbatch/rank 累加 count 和 valid 后求比例，不平均不同分母的比例；报告覆盖完整逻辑 token，排除复制样本/SP replica 的重复计数。

## First divergence

先以同语义边界 hash 定位：embedding → block 输出 → final norm → head logits → scorer。第一失败 block 内逐项比较 norm、q/k/v、position encoding、attention/scan、router ids/weights、experts、combine、projection、residual。

同名模块不一定是同一语义边界。恢复 TP/EP/SP 的逻辑切片，注明 hidden/residual、LSE ln/log2、before/after cast；专家张量按 `(token, expert, slot)` 对齐。找到首个不同输出后，必须证明该 op 的输入相同，再用 captured 输入分别调用两侧实现。

后层可能因舍入重新相等，差异不对层数单调；先粗扫，再细化，别盲目二分。后续 token 的差异可能只是前面状态错误传播，先分析最早位置。forced routing、替换 activation、eager/cache-off 是因果诊断，最终修复须回到原始路径验收。

## Symptoms and next probe

| 现象 | 优先检查 |
|---|---|
| 从首枚回复就不同 | token shift、prefill、head/scorer、权重、raw/processed 语义 |
| 只在长序列或 chunk/page 边界后失败 | KV/SSM/KDA 状态、分段/归约、position 与 pack 边界 |
| sparse spikes 或专家不同 | top-k tie、id/slot 顺序、量化 code/稀疏索引变化 |
| θ0 通过，update 后失败 | WeightIterator 映射、所有 bucket/rank、post-load、scale、cache 失效和版本 |
| no-grad 通过，训练失败 | loss 内直接 head、另一 scorer、compile/recompute、保存的离散选择 |
| 仅 graph 或 prefix hit 失败 | 原进程的 capture 地址、workspace、cache metadata、旧权重状态 |
| loss finite 但 grad Inf/NaN | IS log-ratio 的极值和 exp 溢出，再查 backward；不只看均值 |
| 全部 mismatch 很小但比例接近 1 | 每枚 token 都有末位差，不代表差异幅度大；同输入 op/scorer 比较 |
| 许多采样文本完全相同 | seed/worker RNG 和请求复用；字节对齐不证明采样多样性正确 |
| resume 后首次失败 | checkpoint 参数/状态、首次 export/refit 和派生状态重置 |

## Execution and transport

分别记录 graph/eager、compile/autotune、cache hit/miss/reset、prefill/decode、并行布局、cold/refit/sleep/wakeup。输入评分请求不保证走与在线 decode 相同路径，必须记录调用次数和实际 kernel。

NCCL timeout/actor lost/OOM 是实验未回答，不能标记数值失败或通过。先查每 rank 的阶段、weight-update group、拓扑/transport 和最后 timing；CUDA 异步错误回到真实出错 stream/op。不要因任务慢就停止用户其他作业。

修复记录包含最小输入、最后相同/最早不同边界、same-input 控制、机制、触发条件和原复现的前后结果。若本轮目标只是诊断，在完成定位证据后结束。
