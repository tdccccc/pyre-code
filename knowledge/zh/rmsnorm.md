## 背景

RMSNorm 是 LayerNorm 的简化版，LLaMA、Mistral 等现代大模型都用它。研究发现 LayerNorm 起作用的主要是「缩放到统一尺度」，而「减去均值」贡献不大。RMSNorm 去掉了减均值和偏置 $\beta$，计算更少，效果相当。

## 核心公式

对最后一维的向量 $x = (x_1, \dots, x_D)$，先算**均方根**（Root Mean Square）：

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}
$$

再逐元素缩放：

$$
y_i = \frac{x_i}{\operatorname{RMS}(x)} \cdot w_i
$$

$w$ 是长度为 $D$ 的可学习参数。和 LayerNorm 对照：

$$
\text{LayerNorm: } \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta
\qquad
\text{RMSNorm: } \frac{x}{\sqrt{\overline{x^2} + \epsilon}}\, w
$$

当 $\mu = 0$ 时，$\sigma^2 = \overline{x^2}$，两者（除了 $\beta$）完全一样。

## 形状

`x` 为 `(..., D)`，均方值沿最后一维计算得到 `(..., 1)`（保留维度），广播回 `(..., D)`；`weight` 为 `(D,)`。

## 用到的 PyTorch 操作

- `x.pow(2)` 或 `x * x`：逐元素平方。
- `.mean(dim=-1, keepdim=True)`。
- `torch.sqrt`，或 `torch.rsqrt` 直接得到 $1/\sqrt{\cdot}$ 再相乘。

## 常见坑

- **多减了均值**：那就变回 LayerNorm 了。
- **$\epsilon$ 放错位置**：本题的 $\epsilon$ 在根号里面，加在均方值上。
- **`keepdim` 漏掉**导致广播错位。
- **对 `x` 求了绝对值的均值**：RMS 是先平方、再平均、再开方。
