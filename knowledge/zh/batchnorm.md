## 背景

BatchNorm 和 LayerNorm 一样把激活值归一化到均值 $0$、方差 $1$，区别在于统计的方向：BatchNorm 对**每个特征**，统计整个 batch 里所有样本在这个特征上的均值和方差。它是 CNN 时代的标准组件。

因为统计量依赖 batch，它在训练和推理时行为不同：推理时可能一次只来一个样本，没有 batch 可统计，所以训练期间要顺便维护一份**运行统计量**（running statistics）留给推理用。

## 核心公式

输入 $x$ 的形状是 `(N, D)`。对第 $j$ 个特征：

$$
\mu_j = \frac{1}{N}\sum_{n=1}^{N} x_{n,j},
\qquad
\sigma_j^2 = \frac{1}{N}\sum_{n=1}^{N} (x_{n,j} - \mu_j)^2
$$

$$
y_{n,j} = \gamma_j \cdot \frac{x_{n,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j
$$

**训练模式**：用当前 batch 的 $\mu, \sigma^2$ 归一化，同时用**指数滑动平均**更新运行统计量（$m$ 为 momentum）：

$$
\hat{\mu} \leftarrow (1 - m)\,\hat{\mu} + m\,\mu,
\qquad
\hat{\sigma}^2 \leftarrow (1 - m)\,\hat{\sigma}^2 + m\,\sigma^2
$$

$m = 0.1$ 表示新 batch 占 $10\%$ 的权重，历史占 $90\%$。

**推理模式**：不再计算 batch 统计量，直接用 $\hat{\mu}, \hat{\sigma}^2$ 代替公式里的 $\mu, \sigma^2$，也不更新它们。

## 形状

| 量 | 形状 |
| --- | --- |
| `x`、输出 | `(N, D)` |
| `gamma`、`beta`、`running_mean`、`running_var`、$\mu$、$\sigma^2$ | `(D,)` |

沿 `dim=0`（样本维）求统计量，得到 `(D,)`，再广播回 `(N, D)`。

## 用到的 PyTorch 操作

- `x.mean(dim=0)`、`x.var(dim=0, unbiased=False)`。
- **原地操作**：`t.mul_(a)`、`t.add_(b)`、`t.copy_(v)`，带下划线的方法直接修改 `t` 本身。题目要求 `running_mean` / `running_var` 原地更新，调用方持有的是同一个张量。
- `t.detach()`：把张量从计算图上摘下来，运行统计量不需要梯度。

## 常见坑

- **重新赋值而不是原地更新**：`running_mean = (1 - m) * running_mean + ...` 只是让局部变量指向新张量，调用方传进来的张量没变。
- **推理模式也更新了统计量**，或者推理时仍用 batch 统计量。
- **momentum 方向弄反**：新值权重是 $m$，旧值权重是 $1 - m$。
- **用了无偏方差**归一化：batch 方差用 `unbiased=False`。
