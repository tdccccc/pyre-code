## 背景

**ALiBi**（Attention with Linear Biases）完全不用位置向量，而是在注意力分数上直接加一个**与距离成正比的惩罚**：离得越远，分数扣得越多。它很简单，而且推理时可以处理比训练时更长的序列（外推性好）。BLOOM、MPT 等模型使用它。

## 核心公式

在标准缩放点积分数上加偏置，再做 softmax：

$$
\text{scores}_{h,i,j} = \frac{q_{h,i} \cdot k_{h,j}}{\sqrt{d_h}} \;-\; m_h \cdot |i - j|
$$

每个头 $h$ 有自己的**斜率** $m_h$，按几何级数排列（$H$ 为头数）：

$$
m_h = \frac{1}{2^{8h/H}} = 2^{-8h/H}, \qquad h = 1, 2, \dots, H
$$

例如 $H = 8$ 时，斜率为 $\tfrac12, \tfrac14, \dots, \tfrac{1}{256}$。斜率大的头只关注附近的 token，斜率小的头能看得更远，各头分工覆盖不同距离。

**为什么 softmax 前加偏置等于乘权重**：

$$
\operatorname{softmax}(s_j - m\,d_j) \propto e^{s_j} \cdot e^{-m\, d_j}
$$

即把第 $j$ 个位置原本的注意力权重乘上一个随距离指数衰减的因子，再重新归一化。

## 形状

| 量 | 形状 |
| --- | --- |
| `Q`、`K`、`V` | `(B, S, D)` |
| 拆头后 | `(B, H, S, d_h)`，$d_h = D / H$ |
| 斜率 $m$ | `(H,)` → `(H, 1, 1)` |
| 距离 $\lvert i - j \rvert$ | `(S, S)` |
| 偏置 | `(H, S, S)` → 广播为 `(1, H, S, S)` |
| 分数 | `(B, H, S, S)` |

## 用到的 PyTorch 操作

- `pos.unsqueeze(0) - pos.unsqueeze(1)`：广播得到 $S \times S$ 的差值矩阵；`.abs()` 取绝对值。
- `t.view(H, 1, 1)`、`t.unsqueeze(0)`：调整维度以便广播。
- 拆头、合头同多头注意力：`view` + `transpose` + `reshape`。

## 常见坑

- **$h$ 从 $0$ 开始**：$h = 0$ 会得到斜率 $1$，整组斜率都错位了。
- **符号写反**：偏置是 $-m_h \lvert i-j \rvert$，是惩罚。
- **偏置加在 softmax 之后**：必须加在 softmax 之前的分数上。
- **广播维度对不上**：偏置 `(H, S, S)` 要对齐分数的 `(B, H, S, S)`，在最前面补 batch 维，而不是在中间。
- 本题的距离是对称的 $|i - j|$，没有因果掩码。
