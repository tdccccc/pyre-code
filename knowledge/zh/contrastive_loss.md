## 背景

**对比学习**不需要人工标签，而是利用数据中天然的配对关系：让配对样本的表示**靠近**，不配对的**远离**。

- **CLIP**：图片和它的文字描述是一对，batch 中其他图片的描述都是负样本
- **SimCLR**：同一张图片的两种数据增强是一对，其他图片的增强是负样本

一个 batch 有 $N$ 对，对每个查询 $q_i$，正样本是 $k_i$，其余 $N - 1$ 个键都是负样本，不需要额外采样。

## 核心公式

**相似度**：$q$、$k$ 已经 L2 归一化，所以点积就是余弦相似度，范围 $[-1, 1]$。

**InfoNCE 损失**：

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(q_i \cdot k_i / \tau)}{\sum_{j=1}^{N} \exp(q_i \cdot k_j / \tau)}
$$

**等价于分类**：把 $S_{ij} = q_i \cdot k_j / \tau$ 看成 $N \times N$ 的 logits 矩阵，第 $i$ 行是一个 $N$ 分类问题，正确答案是第 $i$ 类。所以损失就是以 `[0, 1, …, N-1]` 为标签的交叉熵，正样本都在对角线上：

$$
L = \frac{1}{N} \sum_{i} \Bigl(\operatorname{LSE}(S_{i,:}) - S_{ii}\Bigr)
$$

**温度** $\tau$：余弦相似度只在 $[-1, 1]$ 之间，直接做 softmax 太平坦。除以一个小的 $\tau$（如 $0.07$）把差异放大约 $14$ 倍，分布更尖锐，模型被迫更好地区分难负样本。

**基准值**：嵌入随机时所有相似度都接近 $0$，每行 softmax 接近均匀分布，损失 $\approx -\log \frac{1}{N} = \log N$。

CLIP 实际使用**对称**损失：图→文和文→图两个方向各算一次再平均。本题只算 $q \to k$ 一个方向。

## 形状

| 量 | 形状 |
| --- | --- |
| `q`、`k` | `(N, D)` |
| logits $S = q k^{\top} / \tau$ | `(N, N)` |
| 每行的 LSE、对角线 | `(N,)` |
| 输出 | 标量 |

## 用到的 PyTorch 操作

- `q @ k.T`。
- `logits.logsumexp(dim=-1)`，或手写：减去每行最大值后再 `exp`、`sum`、`log`。
- `logits[torch.arange(N), torch.arange(N)]` 或 `torch.diagonal(logits)`：取对角线。

## 常见坑

- **忘了除以温度**。
- **logits 方向**：第 $i$ 行应是 $q_i$ 与所有键的相似度，即 $q k^{\top}$，softmax 沿每行（`dim=-1`）。
- **手写 LSE 不减最大值**：$\tau$ 很小时 logits 可达十几甚至几十，`exp` 容易溢出。
- **用了 `F.cross_entropy`**：本题要求手动实现。
