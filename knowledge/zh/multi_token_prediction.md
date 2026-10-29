## 背景

标准语言模型在每个位置只预测**下一个** token。**多 token 预测**（MTP，Meta 2024；DeepSeek-V3 也使用了类似的做法）在同一个隐藏状态上接 $N$ 个独立的输出头，同时预测接下来的 $N$ 个 token。

好处：

- **训练信号更密**：每个位置提供 $N$ 个监督目标
- **鼓励模型「往前想」**：要预测更远的 token，隐藏状态必须编码更多关于后续内容的规划信息
- **推理加速**：额外的头可以充当草稿模型，配合推测解码一次产出多个 token

## 核心公式

位置 $t$ 的隐藏状态 $h_t$，第 $i$ 个头 $W^{(i)} \in \mathbb{R}^{D \times V}$ 预测第 $t + i$ 个 token：

$$
z^{(i)}_t = h_t\, W^{(i)},
\qquad
\ell^{(i)}_t = -\log \operatorname{softmax}\bigl(z^{(i)}_t\bigr)_{y_{t,i}}
$$

**总损失**：先对所有位置平均，再对 $N$ 个头平均：

$$
L = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{B S} \sum_{b, t} \ell^{(i)}_{b,t}
$$

$N = 1$ 时就是普通的下一个 token 交叉熵。

**稳定的 log-softmax**：

$$
\log \operatorname{softmax}(z)_k = (z_k - m) - \log \sum_j e^{z_j - m},
\qquad m = \max_j z_j
$$

减去最大值后所有指数都 $\le 0$，不会溢出，求和至少为 $1$，对数也不会出问题。

## 形状

| 量 | 形状 |
| --- | --- |
| `hidden_states` | `(B, S, D)` |
| 每个头 `heads[i]` | `(D, V)` |
| 每个头的 logits / log 概率 | `(B, S, V)` |
| `targets` | `(B, S, N)`，`targets[:, :, i]` 是第 $i$ 个头的标签 |
| 取出的目标 log 概率 | `(B, S)` |
| 输出 | 标量 |

## 用到的 PyTorch 操作

- `hidden_states @ heads[i]`。
- `t.max(dim=-1, keepdim=True).values`、`torch.exp`、`torch.log`、`.sum(dim=-1, keepdim=True)`。
- `log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)`：`gather` 要求索引与输入维数相同，所以先在末尾补一维，取完再去掉。

## 常见坑

- **用了 `F.cross_entropy` 或 `F.log_softmax`**：本题要求手写。
- **不减最大值**：logits 大时 `exp` 溢出。
- **标签取错**：第 $i$ 个头用 `targets[:, :, i]`。
- **平均方式**：每个头先对所有位置取均值，再对头取均值。
- `gather` 的索引必须是 `long` 类型。
