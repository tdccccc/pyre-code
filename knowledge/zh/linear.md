## 背景

线性层（全连接层）是神经网络最基本的积木：每个输出都是所有输入的加权和再加一个偏置。注意力里的 Q/K/V 投影、MLP 里的升维降维，全都是线性层。

## 核心公式

对单个输入向量 $x \in \mathbb{R}^{d_{\text{in}}}$：

$$
y = W x + b, \qquad W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}},\ b \in \mathbb{R}^{d_{\text{out}}}
$$

即 $y_j = \sum_{i} W_{j,i}\, x_i + b_j$。实际中输入按行堆叠成矩阵 $X$（每行一个样本），于是写成：

$$
Y = X W^{\top} + b
$$

PyTorch 把权重存成 `(out_features, in_features)`，所以前向要转置。

**初始化**。如果权重方差太大，输出会随层数指数级膨胀；太小则不断衰减。设输入各分量独立、方差为 $1$，则：

$$
\operatorname{Var}(y_j) = d_{\text{in}} \cdot \operatorname{Var}(W_{j,i})
$$

令 $\operatorname{Var}(W) = 1/d_{\text{in}}$，即 $W \sim \mathcal{N}\!\left(0, \tfrac{1}{d_{\text{in}}}\right)$，输出方差就保持在 $1$ 左右。也就是标准正态随机数乘以 $1/\sqrt{d_{\text{in}}}$。偏置初始化为 $0$。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(*, in_features)`，`*` 表示任意多个前导维度 |
| `weight` | `(out_features, in_features)` |
| `bias` | `(out_features,)` |
| 输出 | `(*, out_features)` |

`@` 只作用于最后两维，前导维度原样保留；`bias` 通过广播加到每一行。

## 用到的 PyTorch 操作

- `torch.randn(out, in)`：标准正态随机数。
- `torch.zeros(out)`。
- `t.requires_grad_(True)`，或创建时传 `requires_grad=True`：让参数参与求导。
- `W.T`：二维张量的转置。

## 常见坑

- **权重形状写反**成 `(in, out)`：测试会检查 `weight.shape`。
- **在乘过缩放系数之后的结果上才要求梯度**：先 `randn(...) * scale` 得到新张量，再对它调用 `requires_grad_(True)`。如果先对 `randn` 的结果设 `requires_grad`，再乘系数，存下来的就不是叶子张量了。
- **缩放用了 $1/d_{\text{in}}$**：标准差是 $1/\sqrt{d_{\text{in}}}$，方差才是 $1/d_{\text{in}}$。
