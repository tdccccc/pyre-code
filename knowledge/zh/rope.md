## 背景

正弦位置编码是**加**到输入上的，位置信息要经过线性投影才能进入注意力分数，比较间接。**RoPE**（Rotary Position Embedding）换了思路：在算注意力分数之前，直接把查询和键向量按位置**旋转**一个角度。LLaMA、Qwen、Mistral 等主流大模型都用它。

## 核心公式

**二维旋转**：把向量 $(x_1, x_2)$ 逆时针旋转角度 $\theta$：

$$
R(\theta) \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
=
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
=
\begin{pmatrix} x_1 \cos\theta - x_2 \sin\theta \\ x_1 \sin\theta + x_2 \cos\theta \end{pmatrix}
$$

**RoPE**：把 $D$ 维向量拆成 $D/2$ 对 $(x_{2i}, x_{2i+1})$，位置为 $m$ 的向量，第 $i$ 对旋转角度 $m\,\theta_i$，频率和正弦编码相同：

$$
\theta_i = \frac{1}{10000^{2i/D}}, \qquad i = 0, 1, \dots, \tfrac{D}{2} - 1
$$

**为什么能编码相对位置**：旋转矩阵满足 $R(\alpha)^{\top} R(\beta) = R(\beta - \alpha)$，所以位置 $m$ 的查询和位置 $n$ 的键的点积为

$$
\bigl(R(m\theta)\, q\bigr)^{\top} \bigl(R(n\theta)\, k\bigr) = q^{\top} R\bigl((n - m)\,\theta\bigr)\, k
$$

只依赖相对距离 $n - m$，而与绝对位置无关。

**保持范数**：旋转不改变长度，$\lVert R(\theta)\, x \rVert = \lVert x \rVert$。

## 形状

| 量 | 形状 |
| --- | --- |
| `q`、`k` | `(B, S, D)` |
| 角度 $m\,\theta_i$、$\cos$、$\sin$ | `(S, D/2)`，广播到 `(B, S, D/2)` |
| 偶数分量 `x[..., 0::2]`、奇数分量 `x[..., 1::2]` | `(B, S, D/2)` |
| 旋转后 stack 到最后一维 | `(B, S, D/2, 2)` |
| `flatten(-2)` 拼回 | `(B, S, D)`，偶奇交错 |

## 用到的 PyTorch 操作

- `x[..., 0::2]`、`x[..., 1::2]`：取最后一维的偶数位、奇数位。
- `torch.stack([a, b], dim=-1)`：在最后新增一维，把 $(a_j, b_j)$ 配成一对。
- `t.flatten(-2)`：把最后两维合并，得到 $a_0, b_0, a_1, b_1, \dots$ 的交错顺序。
- `torch.cos`、`torch.sin`、`torch.arange`。

## 常见坑

- **拼回时顺序不对**：`torch.cat([a, b], dim=-1)` 得到的是前一半全是 $a$、后一半全是 $b$，和输入的交错排列不对应。
- **混用另一种约定**：HuggingFace 实现里常见的 `rotate_half` 把前一半和后一半配对，本题按奇偶配对，两者的数值结果不同。
- **只旋转了 q**：q 和 k 都要旋转，点积才只依赖相对位置。
- **旋转方向写反**：第一个分量是 $x_1\cos - x_2\sin$，第二个分量是 $x_1\sin + x_2\cos$。
- 位置从 $0$ 开始。
