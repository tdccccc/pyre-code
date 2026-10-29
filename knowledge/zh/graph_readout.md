## 背景

有些任务要对**整张图**做预测，比如判断一个分子是否有毒。GNN 层输出的是每个节点的表示，还需要一个**读出**（readout，也叫图池化）操作，把一张图所有节点的表示汇总成一个图级向量。

**批处理多张图**：不同的图节点数不同，常见做法是把一个 batch 里所有图的节点拼成一个大矩阵，再用一个 `batch` 向量记录每个节点属于第几张图：

$$
\texttt{batch} = [\underbrace{0, 0, 0}_{\text{图 } 0},\ \underbrace{1, 1}_{\text{图 } 1},\ \underbrace{2, 2, 2, 2}_{\text{图 } 2}]
$$

## 核心公式

记 $\mathcal{V}_g$ 为第 $g$ 张图的节点集合：

$$
\text{sum:}\ \ h_g = \sum_{v \in \mathcal{V}_g} x_v,
\qquad
\text{mean:}\ \ h_g = \frac{1}{|\mathcal{V}_g|} \sum_{v \in \mathcal{V}_g} x_v,
\qquad
\text{max:}\ \ h_{g,k} = \max_{v \in \mathcal{V}_g} x_{v,k}
$$

max 是对每个特征维**分别**取最大值。

**置换不变性**：图的节点没有固有顺序，同一张图换一种节点编号，读出结果必须不变。求和、平均、最大值都只依赖节点的**集合**，满足这个要求。

**表达能力的差别**（见「GIN」一题）：

| 读出 | 能区分 | 不能区分的例子 |
| --- | --- | --- |
| sum | 节点特征的多重集合（含个数） | — |
| mean | 特征的分布（比例） | $\{a\}$ 与 $\{a, a\}$ |
| max | 出现过的最大值 | $\{a, b\}$ 与 $\{a, a, b\}$ |

## 形状

| 量 | 形状 |
| --- | --- |
| `X` | `(total_nodes, D)` |
| `batch` | `(total_nodes,)` |
| 图的数量 | `batch.max() + 1` |
| 每张图的节点数 | `(num_graphs,)` |
| 输出 | `(num_graphs, D)` |

## 用到的 PyTorch 操作

- `out.index_add_(0, batch, X)`：把 `X` 的第 $v$ 行加到 `out` 的第 `batch[v]` 行，一次完成分组求和。
- 节点计数：对全 $1$ 向量做同样的 `index_add_`，或用 `torch.bincount(batch)`。
- `count.unsqueeze(1)`：`(G,)` → `(G, 1)`，才能按行广播相除。
- max：按图循环，`X[batch == g].max(dim=0).values`；或用 `scatter_reduce(..., reduce='amax')`。

## 常见坑

- **图数写成 `batch.max()`**：编号从 $0$ 开始，图数要 $+1$。
- **平均时广播错位**：节点数要变成列向量再除。
- **max 的初始值**：如果先建结果张量再逐步取最大，初始值要用 $-\infty$，不能用 $0$（特征可能全为负）。
- **不支持的模式**应报错。
