## 背景

训练框架里的 LayerNorm 反向通常是手写的融合算子：一个 kernel 直接从 $\mathrm{d}y$、$x$ 算出所有梯度，而不是让 autograd 沿着「减均值 → 求方差 → 开方 → 相除 → 缩放」一步步回传，后者需要保存大量中间结果。

难点在于：均值 $\mu$ 和方差 $\sigma^2$ 依赖于整行的所有元素，所以 $x_j$ 的梯度会受到同一行**所有**输出的影响，不是简单的逐元素链式法则。

## 核心公式

**前向**（沿最后一维，长度 $D$）：

$$
\mu = \frac{1}{D}\sum_i x_i,
\quad
\sigma^2 = \frac{1}{D}\sum_i (x_i - \mu)^2,
\quad
r = \frac{1}{\sqrt{\sigma^2 + \epsilon}},
\quad
\hat{x}_i = (x_i - \mu)\, r,
\quad
y_i = \gamma_i \hat{x}_i + \beta_i
$$

**参数梯度**（对所有前导维度求和，因为同一个 $\gamma, \beta$ 被每一行共享）：

$$
\mathrm{d}\beta = \sum_{\text{行}} \mathrm{d}y,
\qquad
\mathrm{d}\gamma = \sum_{\text{行}} \mathrm{d}y \odot \hat{x},
\qquad
\mathrm{d}\hat{x} = \mathrm{d}y \odot \gamma
$$

**$\hat{x}$ 对 $x$ 的导数**：用到

$$
\frac{\partial \mu}{\partial x_j} = \frac{1}{D},
\qquad
\frac{\partial \sigma^2}{\partial x_j} = \frac{2(x_j - \mu)}{D},
\qquad
\frac{\partial r}{\partial x_j} = -\frac12 (\sigma^2 + \epsilon)^{-3/2} \cdot \frac{2(x_j - \mu)}{D} = -\frac{r^2\, \hat{x}_j}{D}
$$

于是

$$
\frac{\partial \hat{x}_i}{\partial x_j}
= r\left(\delta_{ij} - \frac{1}{D}\right) + (x_i - \mu)\left(-\frac{r^2 \hat{x}_j}{D}\right)
= r\left(\delta_{ij} - \frac{1}{D} - \frac{\hat{x}_i \hat{x}_j}{D}\right)
$$

**输入梯度**：$\mathrm{d}x_j = \sum_i \mathrm{d}\hat{x}_i\, \partial \hat{x}_i / \partial x_j$，三项分别求和：

$$
\mathrm{d}x = r \Bigl(\mathrm{d}\hat{x} - \operatorname{mean}(\mathrm{d}\hat{x}) - \hat{x} \odot \operatorname{mean}(\mathrm{d}\hat{x} \odot \hat{x})\Bigr)
$$

$\operatorname{mean}$ 沿最后一维。

**几何意义**：从 $\mathrm{d}\hat{x}$ 中减掉它沿全 $1$ 方向的分量和沿 $\hat{x}$ 方向的分量。这两个方向正是归一化「抹掉」的自由度：给 $x$ 加常数（均值会吸收）、把 $x$ 整体缩放（方差会吸收）都不改变 $\hat{x}$，所以这两个方向上的梯度必须为 $0$。

## 形状

| 量 | 形状 |
| --- | --- |
| `dy`、`x`、`dx` | `(*, D)` |
| $\mu$、$\sigma^2$、$r$ 及两个 mean | `(*, 1)` |
| `gamma`、`dgamma`、`dbeta` | `(D,)` |

## 用到的 PyTorch 操作

- `.mean(dim=-1, keepdim=True)`、`torch.sqrt` 或 `torch.rsqrt`。
- `t.sum(dim=tuple(range(t.dim() - 1)))`：对除最后一维外的所有维度求和。

## 常见坑

- **只对 `dim=0` 求和**：输入是 3 维 `(B, S, D)` 时，参数梯度要对 `(0, 1)` 两个维度求和。
- **用了无偏方差**：前向的方差除以 $D$。
- **忘了先乘 $\gamma$**：公式里用的是 $\mathrm{d}\hat{x} = \mathrm{d}y \odot \gamma$，而 $\mathrm{d}\gamma$ 用的是原始的 $\mathrm{d}y$。
- **只减均值、漏了投影项**：结果会和 autograd 对不上。
- 自检：$\gamma$ 全为 $1$ 时，每行 $\mathrm{d}x$ 之和为 $0$。
