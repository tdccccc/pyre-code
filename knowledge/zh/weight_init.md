## 背景

如果初始权重太大，信号每经过一层就被放大，深层网络里激活值和梯度会爆炸；太小则逐层衰减直到消失。好的初始化让**每一层输出的方差保持不变**。

Xavier 初始化（$\operatorname{Var}(W) = 1/n$）是为 tanh 这类关于原点对称的激活函数推导的。ReLU 会把一半的输入清零，信号能量减半，所以 Kaiming（He）初始化把方差翻倍。

## 核心公式

线性层 $y_j = \sum_{i=1}^{n} W_{j,i}\, x_i$，$n$ 为输入维度（**fan_in**）。设 $W$ 与 $x$ 独立，$W$ 均值为 $0$：

$$
\operatorname{Var}(y_j) = n \cdot \operatorname{Var}(W) \cdot \mathbb{E}[x_i^2]
$$

如果 $x = \operatorname{ReLU}(h)$，$h$ 关于 $0$ 对称，ReLU 只保留正的一半：

$$
\mathbb{E}[x^2] = \mathbb{E}\bigl[\operatorname{ReLU}(h)^2\bigr] = \tfrac{1}{2}\, \mathbb{E}[h^2] = \tfrac12 \operatorname{Var}(h)
$$

代入得 $\operatorname{Var}(y) = \tfrac{n}{2} \operatorname{Var}(W) \operatorname{Var}(h)$。要让方差逐层不变，需要 $\tfrac{n}{2} \operatorname{Var}(W) = 1$：

$$
\operatorname{Var}(W) = \frac{2}{n_{\text{in}}},
\qquad
W \sim \mathcal{N}\!\left(0,\ \sigma^2\right),\ \ \sigma = \sqrt{\frac{2}{n_{\text{in}}}}
$$

$n_{\text{in}}$ 越小，标准差越大。

## 形状

PyTorch 线性层权重的形状是 `(out_features, in_features)`，所以 $n_{\text{in}}$ = `weight.shape[1]`。

## 用到的 PyTorch 操作

- `t.normal_(mean, std)`：**原地**用正态分布填充。注意第二个参数是**标准差**，不是方差。
- `with torch.no_grad():`：对 `requires_grad=True` 的叶子张量做原地修改时，需要关闭梯度记录，否则会报错。
- `math.sqrt`。

## 常见坑

- **把方差当标准差传**：`normal_(0, 2 / fan_in)` 是错的，要开方。
- **fan_in 取错维度**：`weight.shape[0]` 是 fan_out。
- **返回新张量**：题目要求原地修改并返回同一个张量对象，测试会检查是不是同一个对象。
- **没有 `no_grad`** 就对需要梯度的权重做原地操作，会报 `a leaf Variable that requires grad is being used in an in-place operation`。
