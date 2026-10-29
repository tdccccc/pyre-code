## 背景

softmax 的输出永远是正数，所以即使是毫不相关的 token，也会分到一小部分注意力权重。上下文很长时，这些小权重加起来不可忽视，形成「注意力噪声」，淹没真正相关的信息。

**差分注意力**（Differential Transformer，ICLR 2025）借鉴差分放大器和降噪耳机的原理：算**两张**注意力图，相减。两张图里都有的共模噪声被抵消，真正的信号被保留。

## 核心公式

把 Q、K 沿特征维各切成两半：$Q = [Q_1, Q_2]$，$K = [K_1, K_2]$，每半 $D_h$ 维。

$$
A_1 = \operatorname{softmax}\!\left(\frac{Q_1 K_1^{\top}}{\sqrt{D_h}}\right),
\qquad
A_2 = \operatorname{softmax}\!\left(\frac{Q_2 K_2^{\top}}{\sqrt{D_h}}\right)
$$

$$
\operatorname{DiffAttn}(Q, K, V) = \bigl(A_1 - \lambda A_2\bigr)\, V
$$

**性质**：

- $\lambda = 0$：退化为（只用前一半维度的）标准注意力
- 差分后的权重可以是**负数**，每行之和为 $1 - \lambda$
- 如果两张图完全相同、$\lambda = 1$，输出为 $0$

论文中 $\lambda$ 是可学习的，通过 $\lambda = e^{\lambda_{q1} \cdot \lambda_{k1}} - e^{\lambda_{q2} \cdot \lambda_{k2}} + \lambda_{\text{init}}$ 重参数化，并在之后加 GroupNorm；本题只实现核心的差分部分，$\lambda$ 作为参数传入。

## 形状

| 量 | 形状 |
| --- | --- |
| `Q`、`K` | `(B, S, 2·D_h)` |
| $Q_1, Q_2, K_1, K_2$ | `(B, S, D_h)` |
| $A_1$、$A_2$ | `(B, S, S)` |
| `V`、输出 | `(B, S, D_v)` |

## 用到的 PyTorch 操作

- 切片 `Q[..., :D_h]`、`Q[..., D_h:]`。
- `K.transpose(-2, -1)`、`@`。
- `torch.softmax(t, dim=-1)`（题目不允许用 `F.softmax`）。

## 常见坑

- **缩放因子用了总维度**：是 $\sqrt{D_h}$，即每一半的维度，不是 $\sqrt{2 D_h}$。
- **沿错误的维度切分**：沿最后一维（特征维）切成两半，不是沿序列维。
- **$\lambda$ 乘到了 $A_1$ 上**，或者先相减再 softmax：两次 softmax 分别做完再相减。
- 先合成差分权重，再乘以 $V$（也可以分别乘再相减，结果相同）。
