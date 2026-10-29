## 背景

GCN 给每个邻居的权重是固定的，只由度数决定（$1/\sqrt{d_i d_j}$）。但不同邻居的重要性往往不同。**GAT**（Graph Attention Network）用注意力机制让模型**学习**每条边的权重：根据两端节点的特征，算出节点 $i$ 应该多关注邻居 $j$。

## 核心公式

**特征变换**：$H = X W$，$h_i$ 是 $H$ 的第 $i$ 行。

**注意力分数**：把两个节点的特征拼接，与可学习向量 $a$ 做内积，再过 LeakyReLU：

$$
e_{ij} = \operatorname{LeakyReLU}\bigl(a^{\top} [\, h_i \,\Vert\, h_j \,]\bigr)
$$

$\Vert$ 表示拼接。把 $a$ 拆成前后两半 $a = [a_1 \Vert a_2]$，内积可以分开算：

$$
a^{\top} [\, h_i \,\Vert\, h_j \,] = a_1^{\top} h_i + a_2^{\top} h_j
$$

于是 $N \times N$ 的分数矩阵就是一个列向量加一个行向量的**广播和**：$e = s_1 \mathbf{1}^{\top} + \mathbf{1} s_2^{\top}$，其中 $s_1 = H a_1$、$s_2 = H a_2$。不需要真的构造所有拼接。

**LeakyReLU**（负半轴保留一个小斜率 $\alpha$，通常为 $0.2$）：

$$
\operatorname{LeakyReLU}(x) =
\begin{cases}
x, & x > 0 \\
\alpha x, & x \le 0
\end{cases}
$$

**只在邻居上做 softmax**：非边的位置设为 $-\infty$，再对每行做 softmax：

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})},
\qquad
h'_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, h_j
$$

**与 Transformer 注意力的对比**：

| | GAT | Transformer |
| --- | --- | --- |
| 分数 | 加性：$a_1^{\top} h_i + a_2^{\top} h_j$ | 点积：$q_i^{\top} k_j / \sqrt{d}$ |
| 范围 | 只在图的邻居之间 | 所有位置两两之间 |

## 形状

| 量 | 形状 |
| --- | --- |
| `X` | `(N, F_in)` |
| `W` | `(F_in, F_out)` |
| `H` | `(N, F_out)` |
| `a` | `(2·F_out,)` |
| $s_1$、$s_2$ | `(N,)` → `(N, 1)` 和 `(1, N)` |
| 分数 $e$、注意力 $\alpha$ | `(N, N)` |
| 输出 | `(N, F_out)` |

## 用到的 PyTorch 操作

- `H @ a[:F_out]`：矩阵乘向量，得到 `(N,)`。
- `s1.unsqueeze(1) + s2.unsqueeze(0)`：广播得到 $N \times N$。
- `F.leaky_relu(e, negative_slope)`，或 `torch.where(e > 0, e, slope * e)`。
- `e.masked_fill(A == 0, float('-inf'))`、`torch.softmax(e, dim=-1)`。

## 常见坑

- **$a_1$、$a_2$ 用反了**：$a_1$ 作用在目标节点 $i$（行），$a_2$ 作用在邻居 $j$（列）。
- **softmax 方向**：沿每一行（`dim=-1`）归一化，即对节点 $i$ 的所有邻居。
- **没屏蔽非边**：变成了对所有节点的注意力。
- **忽略 `negative_slope` 参数**：测试会传入自定义值。
- 本题的 $A$ 已包含自环，不需要再加。
