## 背景

softmax 注意力必须先算出 $S \times S$ 的分数矩阵，时间和显存都是 $O(S^2)$。**线性注意力**把相似度函数换成可以拆开的形式，利用矩阵乘法结合律调换计算顺序，复杂度变成关于 $S$ 线性。

## 核心公式

把注意力写成一般形式，$\operatorname{sim}$ 是相似度函数：

$$
o_i = \frac{\sum_j \operatorname{sim}(q_i, k_j)\, v_j}{\sum_j \operatorname{sim}(q_i, k_j)}
$$

softmax 注意力取 $\operatorname{sim}(q, k) = \exp(q^{\top} k / \sqrt{d})$。线性注意力改用一个**特征映射** $\phi$，令 $\operatorname{sim}(q, k) = \phi(q)^{\top} \phi(k)$。这时 $\phi(q_i)$ 与 $j$ 无关，可以提到求和外面：

$$
o_i = \frac{\phi(q_i)^{\top} \overbrace{\textstyle\sum_j \phi(k_j)\, v_j^{\top}}^{\text{所有查询共用}}}{\phi(q_i)^{\top} \underbrace{\textstyle\sum_j \phi(k_j)}_{\text{所有查询共用}}}
$$

写成矩阵形式，关键是**结合律**：

$$
\underbrace{\bigl(\phi(Q)\, \phi(K)^{\top}\bigr)}_{S \times S}\, V
\;=\;
\phi(Q)\, \underbrace{\bigl(\phi(K)^{\top} V\bigr)}_{D_k \times D_v}
$$

左边要 $O(S^2 D)$，右边只要 $O(S\, D_k D_v)$。

**特征映射**：$\phi(x) = \operatorname{elu}(x) + 1$，其中

$$
\operatorname{elu}(x) =
\begin{cases}
x, & x > 0 \\
e^{x} - 1, & x \le 0
\end{cases}
\qquad\Rightarrow\qquad \phi(x) > 0
$$

$\phi$ 恒为正，保证相似度和分母都为正（这是 softmax 的指数原本保证的性质）。

## 形状

| 量 | 形状 |
| --- | --- |
| $\phi(Q)$、$\phi(K)$ | `(B, S, D_k)` |
| $V$ | `(B, S, D_v)` |
| $\phi(K)^{\top} V$ | `(B, D_k, D_v)` |
| 分母中的 $\sum_j \phi(k_j)$ | `(B, 1, D_k)` |
| 分子 $\phi(Q)\,(\phi(K)^{\top} V)$ | `(B, S, D_v)` |
| 分母 | `(B, S, 1)`，广播相除 |

## 用到的 PyTorch 操作

- `F.elu(x) + 1`，或用 `torch.where(x > 0, x, torch.exp(x) - 1) + 1`。
- `torch.bmm`、`t.transpose(1, 2)`。
- `t.sum(dim=1, keepdim=True)`：沿序列维求和。

## 常见坑

- **先算了 $\phi(Q)\,\phi(K)^{\top}$**：结果正确，但又回到了 $O(S^2)$。测试会用长序列检查速度。
- **忘了除以分母**：没有归一化，输出的尺度会随序列长度增长。
- **$\epsilon$ 放错**：加在分母上，$\text{num} / (\text{den} + 10^{-6})$。
- **还用了 $\sqrt{d}$ 缩放或 softmax**：线性注意力里不需要。
