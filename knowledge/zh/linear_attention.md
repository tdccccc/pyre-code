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

## 延伸

- **因果形式就是 RNN**：本题没有因果掩码，每个位置能看到全部。要做因果的，把对 $j$ 的求和限制在 $j \le t$，就得到一组前缀和：

$$
S_t = S_{t-1} + \phi(k_t)\, v_t^{\top},\qquad z_t = z_{t-1} + \phi(k_t)
$$

$$
o_t = \frac{\phi(q_t)^{\top} S_t}{\phi(q_t)^{\top} z_t}
$$

  $S_t$ 是 $D_k \times D_v$ 的矩阵，$z_t$ 是 $D_k$ 维向量。解码时只需要保存这个状态，每步 $O(D_k D_v)$，而且状态大小与已生成的长度无关。这和 softmax 注意力的 KV cache 随 $S$ 线性增长形成对比。
- **训练时**：直接对 $S$ 个位置各存一份 $D_k \times D_v$ 的状态太占显存，实际做法是分块（chunkwise）：块内用并行的矩阵乘，块间传递状态。
- **什么时候划算**：左边的顺序 $O(S^2 D)$，右边 $O(S D_k D_v)$，只有 $S$ 明显大于特征维时才占优。题目里 $S = 2048$、$D = 64$，比值是 $32$；序列比特征维还短时，线性注意力并没有优势。
- **特征映射**：$\operatorname{elu}(x) + 1$ 简单，保证为正。Performer 用随机特征去逼近 softmax 对应的核，其他工作还试过 ReLU 一类。
- **表达力的限制**：见「注意力」专题。后续的工作，如 RetNet、Gated Linear Attention、DeltaNet，在状态更新里加入衰减或门控，让模型可以「遗忘」旧内容，这些形式和 Mamba 一类状态空间模型很接近（见「模型架构」专题）。
- **精度**：分子分母都是对整个序列的累加，半精度下累加应当用 fp32。

## 面试追问

1. **线性注意力为什么能把复杂度降到线性？**
   把相似度写成 $\phi(q)^{\top}\phi(k)$ 之后，$\phi(q_i)$ 与 $j$ 无关，可以提出求和。矩阵乘法有结合律，先算 $\phi(K)^{\top} V$（一个 $D_k \times D_v$ 的小矩阵），再乘 $\phi(Q)$，就避免了 $S \times S$ 的中间矩阵。
2. **为什么特征映射要保证为正？**
   softmax 的指数天然保证权重为正且和为 $1$。线性注意力里，分母是 $\phi(q)^{\top}\sum_j \phi(k_j)$，若 $\phi$ 可以为负，分母可能接近 $0$ 甚至为负，输出会不稳定、权重失去意义。
3. **线性注意力能用于自回归生成吗？**
   可以。因果形式等价于一个 RNN，状态是 $\sum_{j \le t}\phi(k_j) v_j^{\top}$ 和 $\sum_{j \le t}\phi(k_j)$，每步更新一次，解码的时间和显存都与已生成的长度无关。
4. **线性注意力有什么缺点？**
   注意力矩阵的秩不超过 $D_k$，权重也不够尖锐，很难做到把权重几乎全部压到某一个 token 上；定长的状态意味着历史信息被压缩，精确回忆很久之前的内容更难。
5. **线性注意力一定比 softmax 注意力快吗？**
   不一定。它的收益来自 $S \gg D$。序列不长时，$O(S\, D_k D_v)$ 并不比 $O(S^2 D)$ 小多少，而 softmax 注意力有高度优化的 kernel。
