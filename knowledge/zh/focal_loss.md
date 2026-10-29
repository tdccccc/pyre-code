## 背景

目标检测要对图像上成千上万个候选框分类，绝大部分是很容易判断的背景。用交叉熵训练时，每个简单样本的损失虽然小，但数量太多，加起来仍然主导了总损失和梯度，模型把精力都花在已经会了的样本上。

**Focal Loss**（RetinaNet 提出）给交叉熵乘上一个**调制因子**，自动降低简单样本的权重，让训练聚焦在困难样本上。

## 核心公式

记 $p_t$ 为模型给正确类别的概率：$p_t = \operatorname{softmax}(z)_{y}$。交叉熵为 $-\log p_t$。

**Focal Loss**：

$$
\operatorname{FL}(p_t) = -\alpha\, (1 - p_t)^{\gamma} \log p_t
$$

- $(1 - p_t)^{\gamma}$：调制因子。样本越简单（$p_t$ 越接近 $1$），因子越接近 $0$
- $\gamma$：聚焦参数，越大降权越强；$\gamma = 0$ 时退化为 $\alpha$ 加权的交叉熵
- $\alpha$：平衡因子

**$\gamma = 2$ 时调制因子的大小**：

| $p_t$ | $(1 - p_t)^2$ | 相对交叉熵的权重 |
| --- | --- | --- |
| $0.9$（简单） | $0.01$ | 降为 $1/100$ |
| $0.5$ | $0.25$ | 降为 $1/4$ |
| $0.1$（困难） | $0.81$ | 几乎不变 |

**数值稳定的 softmax**：先减去每行最大值再求指数，$\operatorname{softmax}(z)_c = e^{z_c - m} / \sum_k e^{z_k - m}$。

## 形状

| 量 | 形状 |
| --- | --- |
| `logits`、概率 | `(N, C)` |
| `targets`、$p_t$ | `(N,)` |
| 输出 | 标量（均值） |

## 用到的 PyTorch 操作

- `logits.max(dim=-1, keepdim=True).values`、`torch.exp`、`.sum(dim=-1, keepdim=True)`。
- `probs[torch.arange(N), targets]`：每行取出正确类别的概率。
- `torch.log(p_t + 1e-8)`、`**`。

## 常见坑

- **$\log(0)$**：$p_t$ 下溢为 $0$ 时得到 $-\infty$，加一个小 $\epsilon$。
- **调制因子写成 $p_t^{\gamma}$**：应为 $(1 - p_t)^{\gamma}$，否则反而放大了简单样本。
- **不稳定的 softmax**：不减最大值时大 logits 会溢出。
- **成对索引写成 `probs[:, targets]`**：得到 `(N, N)`，不是每行一个值。
- 不能使用 `F.*`；结果对 batch 取均值。
