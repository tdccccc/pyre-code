## 背景

如果初始权重太大，信号每经过一层就被放大，深层网络里激活值和梯度会爆炸；太小则逐层衰减直到消失。好的初始化让**每一层输出的方差保持不变**。

Xavier（Glorot）初始化取 $\operatorname{Var}(W) = 2/(n_{\text{in}} + n_{\text{out}})$，输入输出维度相同时就是 $1/n$，它是为 tanh 这类关于原点对称的激活函数推导的。ReLU 会把一半的输入清零，信号能量减半，所以 Kaiming（He）初始化把方差翻倍。

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

## 延伸

- **Xavier 的推导**：前向要求 $n_{\text{in}} \operatorname{Var}(W) \approx 1$。反向时，梯度 $\partial L/\partial x = W^{\top}\, \partial L/\partial y$ 是对 $n_{\text{out}}$ 个输出求和，要求 $n_{\text{out}} \operatorname{Var}(W) \approx 1$。两条通常不能同时满足，Glorot 与 Bengio（2010）取折中：

$$
\operatorname{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

  他们的分析针对的是 tanh 这类在 $0$ 附近近似线性、关于原点对称的激活函数。
- **Kaiming 只看一边**：`mode='fan_in'` 保持前向的方差（本题用的），`mode='fan_out'` 保持反向的方差。在方阵里两者相同。PyTorch 里对应 `nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')`，`nonlinearity='relu'` 提供的增益是 $\sqrt{2}$。
- **均匀分布也行**：只要方差对，用正态还是均匀分布影响不大。均匀分布 $\mathcal{U}(-a, a)$ 的方差是 $a^2/3$，所以 Kaiming 的均匀版本取 $a = \sqrt{6/n_{\text{in}}}$，Xavier 的取 $a = \sqrt{6/(n_{\text{in}} + n_{\text{out}})}$。
- **为什么要深究**：不做方差保持，信号会随层数指数地放大或缩小。例如用 ReLU 却沿用 $1/n_{\text{in}}$，每层二阶矩减半，$30$ 层后只剩 $2^{-30}$ 左右；换成 $2/n_{\text{in}}$ 就基本保持不变。
- **推导的假设**：权重零均值、与输入独立、各分量独立同分布，且忽略层间的相关性。这些假设在训练开始时近似成立，随训练进行就不再成立，所以初始化只保证**起点**，之后靠归一化来维持尺度。
- **Transformer 的初始化**：GPT-2 一类模型常见做法是权重取标准差 $0.02$ 的正态分布，而写入残差流的投影（注意力的输出投影、FFN 的降维投影）再缩小，大致乘以 $1/\sqrt{2L}$（$L$ 为层数，每层有两个残差分支）。原因是残差流上每个分支都累加一份方差，不缩小的话方差随层数增长。
- **零初始化的用法**：有些结构会故意把最后一层初始化为零，让模型一开始是恒等映射。例如 LoRA 的 $B$ 矩阵（见「量化与高效微调」专题）、DiT 里的 adaLN-Zero（见「扩散与流匹配」专题）。
- **宽度扩展**：µP（maximal update parametrization）等方法进一步规定初始化和学习率如何随模型宽度缩放，使小模型上调好的超参数能迁移到大模型。
- **一个实用的检查**：类别数为 $V$ 的分类或语言模型，初始化后的第一步损失应该接近 $\ln V$（均匀预测）。例如 $V = 50257$ 时约为 $10.8$。远高于这个数，通常说明输出层初始化过大。

## 面试追问

1. **为什么不能把权重全部初始化为零（或同一个常数）？**
   同一层的神经元收到相同的输入、相同的梯度，更新后依然完全相同，永远无法分化（对称性问题）。偏置初始化为零没有问题，因为权重已经打破了对称。
2. **Xavier 和 Kaiming 初始化有什么区别？**
   Xavier 取 $2/(n_{\text{in}} + n_{\text{out}})$，兼顾前向与反向，针对 tanh 等对称激活；Kaiming 取 $2/n_{\text{in}}$，多出的因子 $2$ 补偿 ReLU 清零一半输入造成的能量损失。
3. **推导一下 Kaiming 初始化的方差。**
   $\operatorname{Var}(y) = n_{\text{in}} \operatorname{Var}(W)\, \mathbb{E}[x^2]$。ReLU 后 $\mathbb{E}[x^2] = \tfrac12 \operatorname{Var}(h)$，所以 $\operatorname{Var}(y) = \tfrac{n_{\text{in}}}{2} \operatorname{Var}(W) \operatorname{Var}(h)$，令其等于 $\operatorname{Var}(h)$ 得 $\operatorname{Var}(W) = 2/n_{\text{in}}$。
4. **fan_in 模式和 fan_out 模式怎么选？**
   fan_in 保持前向激活的尺度，fan_out 保持反向梯度的尺度。通常用 fan_in；若更关心梯度的稳定，可以用 fan_out。
5. **有了 BatchNorm / LayerNorm，还需要关心初始化吗？**
   敏感度降低了，因为每层的输出尺度会被拉回，但并非完全不需要。输出层初始化、残差分支的缩放、浅层的梯度大小，仍会影响训练能否顺利起步，尤其是不加归一化或非常深的网络。
