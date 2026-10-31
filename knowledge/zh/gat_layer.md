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

## 延伸

- **多头注意力**：和 Transformer 一样，GAT 并行用 $K$ 组 $(W, a)$。中间层把各头的输出拼接，最后一层通常取平均。本题是单头。
- **注意力的加性形式**：$a_1^\top h_i + a_2^\top h_j$ 把「目标节点」和「邻居」的贡献分开，所以不需要真正构造所有拼接对，$N \times N$ 的分数矩阵靠广播相加得到。真实的稀疏实现里，分数只对边计算，再对每个节点的入边做分组的 softmax。
- **静态注意力问题与 GATv2**：因为 LeakyReLU 是单调的，$e_{ij} = \operatorname{LeakyReLU}(s_{1,i} + s_{2,j})$ 对 $j$ 的排序只取决于 $s_{2,j}$，与目标节点 $i$ 无关，也就是所有节点关注邻居的偏好顺序一样。GATv2 把 $a$ 移到非线性之后，让排序随 $i$ 变化。
- **仍是加权平均**：softmax 后权重和为 $1$，GAT 的聚合和均值一样，数不出邻居个数，表达能力不如 GIN 的求和（见「图神经网络」专题）。
- **自环的作用**：不加自环时，没有邻居的节点整行被掩成 $-\infty$，softmax 会得到 NaN；本题让 $A$ 自带自环，保证每行至少有一个有效位置。
- **代价**：稠密写法是 $O(N^2)$，稀疏写法与边数成正比，注意力本身也带来额外的显存。

## 面试追问

1. **GAT 和 GCN 的区别？**
   GCN 的邻居权重由度数决定，固定不变；GAT 的权重由两端节点的特征算出来，是数据相关的、可学习的。GAT 也不需要事先知道图的全局结构（度数矩阵），只需要局部的邻居。
2. **GAT 的注意力和 Transformer 的注意力有什么区别？**
   打分函数不同：GAT 是加性，Transformer 是缩放点积。范围不同：GAT 只在图的邻居间做 softmax，Transformer 在所有位置间；从这个角度看，Transformer 可以看作完全图上的 GAT。
3. **为什么 softmax 前把非边设成 $-\infty$，而不是 $0$？**
   $\exp(0) = 1$，设成 $0$ 的位置仍然会分到权重；$\exp(-\infty) = 0$ 才会让非边的权重严格为零，并且不影响其余位置的归一化。
4. **多头是怎么合并的？**
   每个头有自己的 $W$、$a$，算出自己的输出。中间层把各头的输出拼接，最后一层取平均再接分类。
5. **GATv2 修复了 GAT 的什么问题？**
   原始 GAT 的注意力是静态的：所有节点对邻居的排序一致。GATv2 把 $a$ 作用到非线性之后，邻居的排序可以依赖目标节点，表达能力更强。
