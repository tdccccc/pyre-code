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

## 延伸

- **在 CNN 里的形状**：输入是 `(N, C, H, W)` 时，沿 `(N, H, W)` 统计，每个通道一组统计量。本题的 `(N, D)` 是它的简化版。
- **推理时折叠进卷积**：推理时 BN 只是一个逐通道的仿射变换 $y = a\,z + b$，其中

$$
a = \frac{\gamma}{\sqrt{\hat{\sigma}^2 + \epsilon}},
\qquad
b = \beta - a\,\hat{\mu}
$$

如果前面是卷积 $z = W * x + c$，可以把 BN 直接并进卷积的参数：$W' = a\,W$（逐输出通道缩放），$c' = a\,c + b$。推理时就完全没有 BN 的开销了。
- **卷积后接 BN 时，卷积不需要 bias**：卷积的 bias 会被 BN 减均值消掉，$\beta$ 已经承担了偏移的作用。
- **小 batch 的问题**：batch 很小时统计量噪声大，效果明显变差。常见解法是换成 GroupNorm，或用 **SyncBN**：多卡训练时用 all-reduce 汇总所有卡的统计量，代价是每个 BN 层都多一次通信。
- **训练 / 推理不一致**：忘记调 `model.eval()` 是经典 bug，推理时会用当前 batch 的统计量，并且继续改写运行统计量。数据分布变化较大时（比如换数据集微调），运行统计量估得不准，推理效果也会下降。
- **正则化效果**：同一 batch 里的样本通过统计量互相影响，相当于给每个样本加了噪声，有轻微的正则化作用。
- **和 PyTorch 的差异**：`nn.BatchNorm` 归一化用有偏方差，但更新 `running_var` 时用的是**无偏**方差。本题为了简化，两处都用有偏方差，以题面为准。

## 面试追问

1. **BatchNorm 训练和推理有什么不同？**
   训练时用当前 batch 的均值方差，并用滑动平均更新运行统计量；推理时直接用运行统计量，不再计算 batch 统计量，也不更新。
2. **BN 怎么折叠进卷积？**
   推理时 BN 是逐通道的仿射变换，把缩放系数乘进卷积权重，把偏移并进卷积 bias，见上面的公式。
3. **batch size 很小时 BN 有什么问题？怎么办？**
   统计量噪声大，训练不稳，推理时运行统计量也估不准。可以换 GroupNorm，或在多卡上用 SyncBN 汇总统计量。
4. **卷积后面接 BN，卷积还要 bias 吗？**
   不需要，bias 会被减均值抵消，偏移由 $\beta$ 负责。
5. **momentum 越大越好还是越小越好？**
   momentum 是新 batch 的权重。越大，运行统计量越跟得上最近的数据，但噪声也越大；越小越平滑，但适应得慢。默认值 $0.1$ 是个折中。
