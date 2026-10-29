## 背景

大模型逐个生成 token 时，每一步都要把全部权重从显存读一遍，GPU 的算力大多闲着。一次前向处理 $1$ 个 token 和处理 $K$ 个 token，耗时差不多。

**推测解码**利用这一点：

1. 用一个小而快的**草稿模型**先连续猜 $K$ 个 token
2. 大的**目标模型**一次前向，同时算出这 $K$ 个位置上它自己的概率
3. 按一个接受规则逐个验证：接受的直接保留，遇到第一个被拒绝的就用修正分布重新采一个，然后结束本轮

关键在于：这个接受规则能保证最终输出的分布**和直接用目标模型采样完全相同**，加速是无损的。

## 核心公式

在某个位置，记目标模型的分布为 $p$，草稿模型为 $q$，草稿提议的 token 为 $x \sim q$。

**接受概率**：

$$
P(\text{接受 } x) = \min\!\left(1,\ \frac{p(x)}{q(x)}\right)
$$

草稿给出的概率不比目标高（$q(x) \le p(x)$）时必定接受；草稿「过度自信」时按比例拒绝。

**拒绝后重新采样**，从残差分布中采：

$$
r(x) = \frac{\max\bigl(0,\ p(x) - q(x)\bigr)}{\sum_{x'} \max\bigl(0,\ p(x') - q(x')\bigr)}
$$

**为什么输出分布恰好是 $p$**：记总接受概率 $\beta = \sum_x q(x) \min\!\bigl(1, \tfrac{p(x)}{q(x)}\bigr) = \sum_x \min\bigl(q(x), p(x)\bigr)$。注意 $\sum_x \max(0, p - q) = 1 - \sum_x \min(p, q) = 1 - \beta$。于是最终输出 $x$ 的概率为

$$
\underbrace{\min\bigl(q(x), p(x)\bigr)}_{\text{提议并被接受}} + \underbrace{(1 - \beta)\cdot \frac{\max\bigl(0, p(x) - q(x)\bigr)}{1 - \beta}}_{\text{被拒绝后重采样}} = \min(q, p) + \max(0, p - q) = p(x)
$$

## 形状

| 量 | 形状 |
| --- | --- |
| `target_probs`、`draft_probs` | `(K, V)`，第 $i$ 行是第 $i$ 个位置的分布 |
| `draft_tokens` | `(K,)` |
| 返回 | Python 列表，长度 $1$ 到 $K$ |

## 用到的 PyTorch 操作

- `torch.rand(1).item()`：$[0, 1)$ 上的均匀随机数，小于接受概率就接受。
- `torch.clamp(t, min=0)`：实现 $\max(0, \cdot)$。
- `torch.multinomial(probs, 1)`：按分布采样。

## 常见坑

- **比值写反**：是 $p_{\text{target}} / p_{\text{draft}}$。
- **拒绝后从 $p$ 重新采样**：必须从残差分布 $r$ 采样，否则输出分布就偏了。
- **拒绝后还继续验证后面的 token**：后面的草稿 token 是以被拒绝的 token 为前提生成的，已经无效，要立刻返回。
- **残差没有归一化**：`multinomial` 虽然能接受未归一化的权重，但要处理残差全为 $0$ 的情况（此时 $p = q$，不会发生拒绝，但写代码时要防止除以零）。
- 真实实现中全部接受时还会从目标模型多采一个「奖励 token」；本题不要求，全部接受时返回 $K$ 个。
