## 背景

训练框架里的 LayerNorm 反向通常是手写的融合算子：一个 kernel 直接从 $\mathrm{d}y$、$x$ 算出所有梯度，而不是让 autograd 沿着「减均值 → 求方差 → 开方 → 相除 → 缩放」一步步回传，后者需要保存大量中间结果。

难点在于：均值 $\mu$ 和方差 $\sigma^2$ 依赖于整行的所有元素，所以 $x_j$ 的梯度会受到同一行**所有**输出的影响，不是简单的逐元素链式法则。

## 核心公式

**前向**（沿最后一维，长度 $D$）：

$$
\mu = \frac{1}{D}\sum_i x_i,
\qquad
\sigma^2 = \frac{1}{D}\sum_i (x_i - \mu)^2,
\qquad
r = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
\hat{x}_i = (x_i - \mu)\, r,
\qquad
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
\frac{\partial \sigma^2}{\partial x_j} = \frac{2(x_j - \mu)}{D}
$$

$$
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

## 延伸

- **保存什么**：融合实现通常保存输入 $x$（或输出）和每行的 $\mu$、$r$，两个统计量各是 `(*,)` 的小张量；反向里 $\hat{x}$ 用它们廉价地重算。autograd 沿着拆开的运算回传，则要保存好几份与 $x$ 同形状的中间张量。本题只传入 $x$ 和 $\gamma$，统计量在反向里重新算出。
- **两个行内归约**：$\operatorname{mean}(\mathrm{d}\hat{x})$ 和 $\operatorname{mean}(\mathrm{d}\hat{x}\odot\hat{x})$ 可以在同一遍扫描里一起求出，然后一个逐元素式子给出 $\mathrm{d}x$。
- **跨行归约**：$\mathrm{d}\gamma$、$\mathrm{d}\beta$ 要把所有行累加起来，行分散在不同的线程块里，实现时常见的做法是先各块求部分和，再做一次第二级归约。这部分往往比 $\mathrm{d}x$ 更麻烦。
- **关于 $\epsilon$**：上面「几何意义」里的正交性在 $\epsilon$ 可忽略时严格成立。$\epsilon > 0$ 时 $\hat{x}$ 的均方略小于 $1$，$\mathrm{d}x$ 与 $\hat{x}$ 只是近似正交；但推导中没有做任何近似，公式本身仍然是精确的。
- **精度**：统计量和归约在 fp32 下做，哪怕输入输出是 fp16 / bf16。
- **同族的写法**：RMSNorm 的反向去掉均值项；BatchNorm 的形式相同、但均值沿 batch 方向。对比见「手写反向传播」专题。

## 面试追问

1. **$\mathrm{d}x$ 公式里的三项分别来自哪里？**
   $x_j$ 影响 $\hat{x}_i$ 有三条路：直接的 $x_j - \mu$ 里的 $x_j$（第一项 $\mathrm{d}\hat{x}$）、经过均值 $\mu$（第二项，减去 $\mathrm{d}\hat{x}$ 的均值）、经过方差（第三项，减去 $\mathrm{d}\hat{x}$ 在 $\hat{x}$ 方向的投影）。
2. **为什么 $\gamma$ 全为 $1$ 时，每行 $\mathrm{d}x$ 之和为 $0$？**
   LayerNorm 对平移不变，所有 $x_j$ 同加常数，输出不变，所以梯度在全 $1$ 方向的分量为 $0$。代入公式：$\mathrm{d}\hat{x}$ 与它的均值项求和后相消，第三项含 $\sum_j \hat{x}_j = 0$，所以每行之和恰好为 $0$。
3. **为什么 $\mathrm{d}\gamma$、$\mathrm{d}\beta$ 要对 $B$ 和 $S$ 都求和？**
   $\gamma$、$\beta$ 被每个 token 共享，前向相当于把它们广播到 `(B, S, D)`；反向就要把所有位置的梯度加回 `(D,)`。只对第 $0$ 维求和是常见的错误。
4. **为什么 $\mathrm{d}\gamma$ 用 $\mathrm{d}y$，而 $\mathrm{d}x$ 要用 $\mathrm{d}\hat{x} = \mathrm{d}y \odot \gamma$？**
   $y = \gamma \hat{x} + \beta$，$\gamma$ 对 $y$ 的导数是 $\hat{x}$，用原始的 $\mathrm{d}y$；而 $x$ 是经 $\hat{x}$ 才影响 $y$，要先过 $\gamma$ 这一步，所以用 $\mathrm{d}y \odot \gamma$。
5. **手写融合反向相比 autograd 好在哪？**
   省显存：只保存 $x$ 和每行两个统计量，而不是整条链上的中间张量。省访存：LayerNorm 是访存受限的，融合后 $\mathrm{d}y$、$x$ 各读一次、$\mathrm{d}x$ 写一次。代价是要自己保证公式正确，用 float64 的 `gradcheck` 验证。
