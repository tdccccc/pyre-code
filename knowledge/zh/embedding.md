## 背景

模型的输入是 token 的整数编号（词表里的位置），但神经网络需要的是实数向量。**嵌入层**为词表里每个 token 准备一个可学习的向量，前向时按编号把对应的向量取出来。它是语言模型的第一层，GPT 的输出层也常常和它共享同一个矩阵（权重绑定）。

## 核心公式

嵌入矩阵 $W \in \mathbb{R}^{V \times d}$，第 $i$ 行是 token $i$ 的向量（$V$ 为词表大小，$d$ 为嵌入维度）。查表就是取一行：

$$
\operatorname{Embed}(i) = W_{i,:}
$$

它等价于 one-hot 向量乘矩阵：

$$
\operatorname{Embed}(i) = e_i^{\top} W,
\qquad e_i = (0, \dots, 0, \underset{\text{第 } i \text{ 位}}{1}, 0, \dots, 0)
$$

所以嵌入层本质上就是一个没有偏置、输入为 one-hot 的线性层。直接索引省掉了和大量 $0$ 相乘的开销。

**梯度**：由 one-hot 形式可知，$\frac{\partial L}{\partial W_{j,:}}$ 只在被查到的行 $j$ 上非零；同一个编号在一个 batch 里出现多次，它的梯度会**累加**。

## 形状

| 量 | 形状 |
| --- | --- |
| `weight` | `(num_embeddings, embedding_dim)` |
| `indices` | 任意形状，如 `(B, S)`，整数类型 |
| 输出 | `indices.shape + (embedding_dim,)`，如 `(B, S, d)` |

## 用到的 PyTorch 操作

- `nn.Parameter(tensor)`：把张量包装成参数，赋值给 `nn.Module` 的属性后会自动注册，`model.parameters()` 能找到它，并且默认 `requires_grad=True`。
- **高级索引**：`W[indices]` 用一个整数张量去索引第 0 维，结果形状是「索引的形状 + 剩余维度」。
- `torch.randn(V, d)`：初始化。

## 常见坑

- **用普通张量当权重**：不包成 `nn.Parameter`，优化器找不到它，也不会参与训练。
- **形状写反**成 `(embedding_dim, num_embeddings)`：测试会检查 `weight.shape`。
- **把输入展平后忘了还原形状**：直接 `W[indices]` 就能处理任意形状。
- 索引必须是整数类型（`long` / `int`）。
