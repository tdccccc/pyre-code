## 背景

深层网络里，每一层输入的数值范围会随着训练不断漂移，让训练变得不稳定。**归一化**把激活值拉回均值 $0$、方差 $1$ 附近，再交给可学习的参数去缩放和平移。

LayerNorm 对**每个样本（每个 token）自己的特征向量**做归一化，和 batch 里有多少样本无关，所以是 Transformer 的标准选择。

## 核心公式

设最后一维的特征向量为 $x = (x_1, \dots, x_D)$：

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i,
\qquad
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}},
\qquad
y_i = \gamma_i\, \hat{x}_i + \beta_i
$$

- $\epsilon$：一个很小的数（如 $10^{-5}$），防止方差为 $0$ 时除以零
- $\gamma, \beta$：长度为 $D$ 的可学习参数，让网络在需要时恢复原来的尺度和偏移

注意方差除以 $D$，是**有偏估计**；统计学里常见的无偏估计除以 $D - 1$。

## 形状

`x` 的形状是 `(..., D)`，统计量只沿最后一维算：$\mu$、$\sigma^2$ 的形状是 `(..., 1)`，再通过**广播**和 `x` 逐元素运算。`gamma`、`beta` 的形状是 `(D,)`，同样靠广播作用到每个位置。

## 用到的 PyTorch 操作

- `x.mean(dim=-1, keepdim=True)`：沿最后一维求均值，`keepdim=True` 保留那一维（长度变成 1），方便后面广播。
- `x.var(dim=-1, keepdim=True, unbiased=False)`：有偏方差。
- `torch.sqrt`、`torch.rsqrt`（$1/\sqrt{\cdot}$）。

## 常见坑

- **用了无偏方差**：`x.var()` 默认 `unbiased=True`（除以 $D-1$），和 `F.layer_norm` 对不上。
- **`keepdim` 忘了加**：`(B, S, D)` 的均值变成 `(B, S)`，再和 `(B, S, D)` 相减时广播会对错维度，或者直接报错。
- **$\epsilon$ 位置放错**：$\epsilon$ 加在根号里面 $\sqrt{\sigma^2 + \epsilon}$，不是 $\sqrt{\sigma^2} + \epsilon$。
- **归一化维度不对**：LayerNorm 沿特征维（最后一维）；沿 batch 维（`dim=0`）就成了 BatchNorm。
