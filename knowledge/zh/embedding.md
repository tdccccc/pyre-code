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

## 延伸

- **和 Linear 的关系**：Embedding 是输入为 one-hot 的无偏置 Linear。反过来，语言模型的输出头 $\text{logits} = h\, E^{\top}$ 是一个 $d \to V$ 的 Linear，权重形状和输入 Embedding 相同，都是 `(V, d)`。
- **权重绑定（weight tying）**：让输入嵌入和输出头共用同一个矩阵 $E$。
  - 好处：省下 $V \cdot d$ 个参数。以 GPT-2 small 为例，词表 $50257$、$d = 768$，这个矩阵有约 $3.86 \times 10^7$ 个参数，约占总参数量（约 $1.24 \times 10^8$）的三成。
  - 做法在 Press 与 Wolf（2017）等工作中被系统研究；原始 Transformer 论文也在嵌入层与输出层之间共享了权重；GPT-2 同样绑定。
  - 代价：输入和输出共用一套表示，被迫兼顾两种用途。模型越大，$V \cdot d$ 占总参数的比例越小，绑定的收益越小，不少大模型选择不绑定。
  - 绑定后 $E$ 的梯度是输入侧稀疏梯度与输出侧稠密梯度之和。
- **梯度是稀疏的**：一个 batch 里只有出现过的 token 对应的行有梯度，其余行为零（本题的梯度测试检查的就是这一点）。PyTorch 的 `nn.Embedding(..., sparse=True)` 可以产生稀疏梯度，但只有部分优化器支持。输出头的梯度则是稠密的，因为 softmax 对词表里每个 token 都有概率。
- **初始化**：`nn.Embedding` 默认从 $\mathcal{N}(0, 1)$ 采样；GPT-2 一类模型用的是标准差 $0.02$ 的正态分布。初始化的尺度要和后面的归一化、绑定后的输出头一起考虑，否则初始 logits 过大，初始损失会远高于 $\ln V$。
- **嵌入缩放**：原始 Transformer 把嵌入输出乘以 $\sqrt{d_{\text{model}}}$，之后再与位置编码相加（见「位置编码」专题）。常见的解释是让嵌入的尺度与位置编码相当，并与共享权重的初始化尺度相匹配。
- **词表规模的权衡**：词表大，同样的文本被切成更少的 token，序列更短；但 Embedding 和输出头的参数变多，每个 token 的 logits 计算量 $2\, d\, V$ 也变大，需要的训练数据也更多才能把稀有词学好。常把词表补到 $64$ 或 $128$ 的倍数，对齐 GPU 的矩阵乘。
- **输出端的内存**：logits 形状是 `(B, S, V)`，词表大时是显存大头，可以分块计算交叉熵避免整体物化（见「损失函数」专题的「分块 LM Head 交叉熵」一题）。
- **分布式**：词表很大时，Embedding 沿词表维切分到多张卡（见「分布式训练」专题）：每张卡只负责一段 token 编号，不属于自己的编号输出 $0$，最后一次 all-reduce 把结果汇总。
- **位置嵌入**：GPT-2 的可学习位置编码也是一个 Embedding，索引是位置而不是 token 编号。

## 面试追问

1. **Embedding 层和 Linear 层是什么关系？为什么用查表？**
   Embedding 等价于 one-hot 向量乘权重矩阵，也就是一个无偏置 Linear。直接取行把计算量从 $O(V d)$ 降到 $O(d)$，也不用物化 one-hot 向量。
2. **什么是权重绑定？有什么利弊？**
   输入嵌入和输出头共享同一个 `(V, d)` 矩阵。省 $V \cdot d$ 个参数、让输入输出表示一致；代价是限制了灵活性，模型大了之后收益很小。
3. **Embedding 的梯度有什么特点？**
   稀疏：只有 batch 里出现过的 token 对应的行有非零梯度；同一个 token 出现多次，梯度累加。
4. **词表大小怎么权衡？**
   词表大 → 序列更短、推理步数更少；但 Embedding / 输出头参数更多、logits 计算和显存更大，低频词也更难训练。反之亦然。
5. **输入的索引越界会怎样？**
   超出 `[0, num_embeddings)` 的索引会报错（CPU 上是明确的 index 错误，GPU 上常表现为设备端断言失败）。常见原因是分词器词表大小和模型的 `num_embeddings` 对不上。
