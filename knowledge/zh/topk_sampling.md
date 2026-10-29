## 背景

语言模型每一步输出词表上的一个概率分布，下一个 token 怎么选？

- **贪心**（取 argmax）：确定但单调，容易重复。
- **直接按分布采样**：多样，但词表有几万个 token，概率很低的「长尾」加起来不可忽视，偶尔会采到离谱的词。

常用做法是先用**温度**调整分布的尖锐程度，再用 **top-k** 或 **top-p** 砍掉长尾，最后在剩下的 token 中采样。

## 核心公式

**温度** $T$：

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

- $T < 1$：分布更尖锐，$T \to 0$ 时趋向 argmax
- $T = 1$：原分布
- $T > 1$：分布更平坦，$T \to \infty$ 时趋向均匀分布

**top-k**：只保留 logits 最大的 $k$ 个 token，其余设为 $-\infty$（概率为 $0$）。$k = 1$ 即贪心。

**top-p（核采样）**：把 token 按概率从大到小排序为 $p_{(1)} \ge p_{(2)} \ge \dots$，保留**最小的**前缀集合，使其累积概率达到 $p$：

$$
\text{保留第 } r \text{ 名} \iff \sum_{s < r} p_{(s)} < p
$$

即只要排在它**前面**的 token 的概率之和还没到 $p$，它就保留。概率第一的 token 永远保留。和 top-k 相比，top-p 的保留个数会随分布形状自适应：分布很集中时只剩几个，很平坦时保留很多。

过滤后对剩下的 logits 重新做 softmax，概率自动重新归一化。

## 形状

`logits` 为 `(V,)`。排序后得到排好序的值和原始下标 `sorted_idx`，过滤在排序后的顺序上做，然后要按 `sorted_idx` 放回原来的位置。

## 用到的 PyTorch 操作

- `logits.topk(k).values[-1]`：第 $k$ 大的值，作为阈值。
- `torch.sort(t, descending=True)`：返回 `(values, indices)`。
- `torch.cumsum(p, dim=-1)`：累积和。
- `torch.empty_like(t).scatter_(0, sorted_idx, sorted_vals)`：把排序后的值写回原始位置。
- `torch.multinomial(probs, 1)`：按概率采样一个下标；`.item()` 转成 Python 整数。

## 常见坑

- **顺序不对**：先温度，再 top-k，再 top-p。
- **top-p 没有排序**：累积概率必须按从大到小的顺序计算。
- **过滤后没有放回原位**：在排序空间里采到的是名次，不是 token 编号。
- **把概率第一的 token 也过滤掉了**：如果单个最大概率就超过 $p$，用 `cumsum > p` 作条件会把它也去掉。正确条件看的是「它前面的累积和」。
- **返回张量**：题目要求返回 `int`。
- 可以把 $-\infty$ 直接留在 logits 里交给 softmax：$e^{-\infty} = 0$。
