## 背景

GCN 依赖整张图的归一化邻接矩阵，属于**直推式**（transductive）学习：训练和预测都在同一张固定的图上，新加入的节点或新的图需要重新处理整张图。超大图（数十亿节点）还根本放不进显存。

**GraphSAGE**（SAmple and aggreGatE）学的是一个**聚合函数**，而不是每个节点的固定嵌入，所以能直接用到训练时没见过的节点和图（**归纳式**，inductive）。它还对邻居做**采样**：每个节点最多取 $k$ 个邻居，计算量与节点度数无关，可以在大图上做小批量训练（Pinterest 的推荐系统就用了它）。

## 核心公式

**邻居采样**：节点 $i$ 的邻居集合 $\mathcal{N}(i)$，若 $|\mathcal{N}(i)| > k$，随机抽取 $k$ 个，得到 $\mathcal{S}(i)$；否则全部使用。

**均值聚合**：

$$
h_{\mathcal{N}(i)} = \frac{1}{|\mathcal{S}(i)|} \sum_{j \in \mathcal{S}(i)} x_j
$$

没有邻居的节点取零向量。

**拼接 + 变换**：

$$
h_i = \operatorname{ReLU}\bigl([\, x_i \,\Vert\, h_{\mathcal{N}(i)} \,]\, W\bigr)
$$

和 GCN 把自己与邻居混在一起求和不同，拼接让自身特征和邻居信息分别进入线性层，模型可以给它们不同的权重。

**L2 归一化**：

$$
h_i \leftarrow \frac{h_i}{\max\bigl(\lVert h_i \rVert_2,\ 10^{-8}\bigr)}
$$

所有节点嵌入都落在单位球面上，内积就等于余弦相似度。

## 形状

| 量 | 形状 |
| --- | --- |
| `A` | `(N, N)` |
| `X`、邻居均值 | `(N, F_in)` |
| 拼接 | `(N, 2·F_in)` |
| `W` | `(2·F_in, F_out)` |
| 输出 | `(N, F_out)`，每行范数为 $1$（或全零行） |

## 用到的 PyTorch 操作

- `(A[i] > 0).nonzero(as_tuple=True)[0]`：节点 $i$ 的邻居下标。
- `torch.manual_seed(42)`、`torch.randperm(n)[:k]`：随机打乱后取前 $k$ 个，即无放回抽样。
- `X[idx].mean(dim=0)`、`torch.stack`、`torch.cat([X, agg], dim=1)`。
- `t.norm(dim=-1, keepdim=True).clamp(min=1e-8)`。

## 常见坑

- **种子只设一次**：题目要求**每次** `randperm` 之前都调用 `torch.manual_seed(42)`，否则结果和参考答案不同。
- **邻居数不超过 $k$ 时也采样**：只有超过 $k$ 才采样。
- **孤立节点**：没有邻居时，均值是空集的平均，会得到 NaN。要直接用零向量。
- **拼接顺序**：$[x_i, h_{\mathcal{N}(i)}]$，自身在前。
- **归一化时除以零**：全零行的范数为 $0$，要给范数设下限。
