## 背景

语言模型每一步输出一个覆盖整个词表的分布。后训练里几乎所有损失都要用到同一个量：模型给**实际出现的那个 token** 打出的对数概率 $\log \pi(y_t \mid y_{<t})$。

- **SFT**：损失就是它取负再平均。
- **DPO**：把回答里所有 token 的对数概率加起来，得到整条回答的 $\log \pi(y \mid x)$。
- **PPO / GRPO**：逐 token 计算新旧策略的比率 $\exp(\log \pi_{\text{new}} - \log \pi_{\text{old}})$。

所以这一步写错，后面的损失全都跟着错，而且往往不会报错，只是训练效果莫名其妙变差。

## 核心公式

模型在 `input_ids` 上前向一次得到 logits。因果语言模型在位置 $t$ 只看过前 $t+1$ 个 token，它的输出是对**第 $t+1$ 个 token** 的预测：

$$
\log \pi(x_{t+1} \mid x_{\le t}) = z_{t, x_{t+1}} - \log \sum_{v=1}^{V} e^{z_{t, v}}
$$

- $z_t = \text{logits}[t] / \tau$：除以采样温度 $\tau$ 之后的 logits，长度为词表大小 $V$。
- $x_{t+1}$：下一个 token 的 id。
- 第二项是 $\operatorname{logsumexp}(z_t)$，也就是 softmax 的归一化常数取对数。

只有当目标 $x_{t+1}$ 是回答 token 时才保留，否则置 0。

## 形状

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `logits` | (B, T, V) | 去掉最后一个位置后是 (B, T−1, V)，因为最后一个位置没有下一个 token |
| `input_ids` | (B, T) | 去掉第一个 token 后是目标 (B, T−1)，因为第一个 token 没有前文 |
| `response_mask` | (B, T) | 同样取 `[:, 1:]`，跟目标对齐 |
| 输出 | (B, T−1) | 第 $t$ 列对应第 $t+1$ 个 token |

## 用到的 PyTorch 操作

- `x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)`：在最后一维按 id 取值，(B, T−1, V) 取出 (B, T−1)。
- `torch.logsumexp(x, dim=-1)`：先减最大值再求和取对数，大数值下也不会溢出。
- `t.masked_fill(~mask, 0.0)`：把不需要的位置填成 0，不会产生 `-0.0`。

## 常见坑

- **没有错位**：用 `logits[:, t]` 去给 `input_ids[:, t]` 打分。这等于让模型「偷看」自己要预测的 token。
- **掩码错位**：掩码要看**目标** token 的位置，即 `response_mask[:, 1:]`。用 `[:, :-1]` 会把「最后一个提示词 token 预测第一个回答 token」漏掉，反而多算一个回答之后的位置。
- **`log(softmax(x))`**：softmax 对很小的概率会下溢成 0，取对数就成了 $-\infty$，再乘以 0 的掩码变成 NaN。要么用 `log_softmax`，要么像本题一样用 logsumexp 的形式。
- **温度放错位置**：温度要在 softmax **之前**除到 logits 上。把 `log_softmax` 的结果除以 $\tau$，得到的已经不是一个概率分布的对数了。

## 延伸

- **为什么要带温度**：RL 训练里，回答是推理引擎（vLLM、SGLang）用某个温度采出来的，旧策略的对数概率也要在同一个温度下算，比率 $\pi_{\text{new}} / \pi_{\text{old}}$ 才有意义。verl 在计算对数概率前会先 `logits.div_(temperature)`。
- **显存**：完整的 `log_softmax` 会产生一个 (B, T, V) 的新张量。词表十几万、序列几千时这非常大，所以 verl、OpenRLHF 只取目标位置的值，Liger-kernel 等融合 kernel 更是直接分块计算，不保留整张 logits（见「分块 LM Head 交叉熵」一题）。
- **只前向回答部分**：很多框架只保留回答对应的 logits 切片（`logits[:, -response_len-1:-1]`），既省显存，又自然完成了错位。
- **训练和推理的数值不一致**：推理引擎和训练框架用的 kernel、精度不同，同一条回答算出的对数概率会有细微差别。这会让「同一个策略」的比率不严格等于 1，2025 年后不少工作专门修正这个偏差，见「强化学习对齐」专题。

## 面试追问

1. **为什么 logits 和 labels 要错开一位？**
   因果模型在位置 $t$ 只能看到 $x_{\le t}$，它输出的是对 $x_{t+1}$ 的预测。HF 的 `CausalLM` 损失内部就做了 `logits[..., :-1]` 和 `labels[..., 1:]` 的对齐。
2. **DPO 用的是对数概率的和还是平均？**
   原始 DPO 用和，即整条回答的 $\log \pi(y \mid x)$。SimPO 改用按长度平均的对数概率，以减轻对长回答的偏好。
3. **为什么不直接存概率而存对数概率？**
   概率的乘积很快下溢；对数概率相加不会。而且比率可以写成对数概率之差再取指数，数值更稳。
4. **怎样在不实例化 (B, T, V) 的 log_softmax 的情况下算出结果？**
   只需要目标位置的 logit 和每个位置的 logsumexp。两者都可以分块或在融合 kernel 里直接计算，峰值显存只取决于块的大小。
