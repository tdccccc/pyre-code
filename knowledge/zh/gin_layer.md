## 背景

GNN 能区分多少种不同的图？**GIN**（Graph Isomorphism Network）论文证明：消息传递 GNN 区分图结构的能力，最多和经典的 **WL 图同构测试**（Weisfeiler-Lehman）一样强。要达到这个上限，每个节点对邻居的聚合函数必须是**单射**：不同的邻居特征多重集合（允许重复的集合）要映射到不同的结果。

## 核心公式

**为什么要用求和**：常见的聚合方式中，只有求和能区分所有的多重集合：

| 聚合 | 邻居 $\{a\}$ | 邻居 $\{a, a\}$ | 能区分吗 |
| --- | --- | --- | --- |
| mean | $a$ | $a$ | 不能 |
| max | $a$ | $a$ | 不能 |
| sum | $a$ | $2a$ | 能 |

mean 只看比例，max 只看出现过哪些值，都丢掉了「个数」信息。

**GIN 层**：

$$
h'_v = \operatorname{MLP}\Bigl((1 + \epsilon) \cdot h_v + \sum_{u \in \mathcal{N}(v)} h_u\Bigr)
$$

- 邻居部分直接**求和**，不做归一化
- 自身特征乘以 $(1 + \epsilon)$，$\epsilon$ 可学习。这让「自己」和「邻居」的贡献可以被区分开：否则一个节点自己的特征为 $a$、邻居为 $b$，和自己为 $b$、邻居为 $a$ 就完全一样了
- **MLP** 而不是单个线性层：需要足够强的函数（MLP 是万能近似器）来把这个和映射成单射的表示

本题的 MLP：

$$
\operatorname{MLP}(z) = \operatorname{ReLU}(z W_1 + b_1)\, W_2 + b_2
$$

**矩阵形式**：$\sum_{u \in \mathcal{N}(v)} h_u$ 对所有节点一起算就是 $A X$，所以

$$
H' = \operatorname{MLP}\bigl((1 + \epsilon) X + A X\bigr)
$$

## 形状

| 量 | 形状 |
| --- | --- |
| `A` | `(N, N)` |
| `X`、聚合结果 | `(N, F_in)` |
| `W1`、`b1` | `(F_in, F_hidden)`、`(F_hidden,)` |
| `W2`、`b2` | `(F_hidden, F_out)`、`(F_out,)` |
| 输出 | `(N, F_out)` |

## 用到的 PyTorch 操作

- `A @ X`：邻居求和。
- `torch.relu`、`@`、加偏置（广播）。

## 常见坑

- **对 $A$ 做了归一化**（像 GCN 那样）：GIN 的关键就是不归一化的求和。
- **又加了自环**：自身已经由 $(1 + \epsilon) X$ 计入，再用 $A + I$ 就重复算了一次。
- **第二层后面也加了 ReLU**：本题的 MLP 最后一层没有激活。
- **$\epsilon$ 当成 Python 数处理**：它是一个张量（可学习），直接参与运算即可，梯度会传给它。
