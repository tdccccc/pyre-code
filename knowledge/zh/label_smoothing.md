## 背景

用 one-hot 标签训练时，交叉熵要求 $p_y \to 1$，这只有在正确类别的 logit 比其他类别高出无穷多时才能达到。模型于是会不断放大 logits，变得过度自信，泛化变差。

**标签平滑**把一小部分概率 $\varepsilon$ 从正确类别「匀」给其他类别，目标分布不再是 one-hot，最优的 logits 就是有限的。原始 Transformer 用 $\varepsilon = 0.1$。

## 核心公式

$C$ 个类别，正确类别 $y$，平滑后的目标分布：

$$
q_c =
\begin{cases}
1 - \varepsilon, & c = y \\[1mm]
\dfrac{\varepsilon}{C - 1}, & c \ne y
\end{cases}
\qquad \sum_c q_c = 1
$$

损失是目标分布与预测分布的交叉熵：

$$
\ell = -\sum_{c=1}^{C} q_c \log p_c
= (1 - \varepsilon)\bigl(-\log p_y\bigr) + \frac{\varepsilon}{C - 1} \sum_{c \ne y} \bigl(-\log p_c\bigr)
$$

$\varepsilon = 0$ 时 $q$ 就是 one-hot，退化为普通交叉熵。

**最优解**：交叉熵在 $p = q$ 时最小，所以模型被鼓励输出 $p_y = 1 - \varepsilon$，而不是 $1$，对应的 logit 差是有限值：

$$
z_y - z_c = \log \frac{1 - \varepsilon}{\varepsilon / (C - 1)}
$$

$\log p$ 用稳定的 log-softmax 计算：$\log p_c = z_c - \operatorname{LSE}(z)$，$\operatorname{LSE}(z) = m + \log \sum_k e^{z_k - m}$。

## 形状

| 量 | 形状 |
| --- | --- |
| `logits`、$\log p$、$q$ | `(N, C)` |
| `targets` | `(N,)` |
| 每个样本的损失 | `(N,)` |
| 输出 | 标量（均值） |

## 用到的 PyTorch 操作

- `torch.full_like(t, value)`：同形状、填满某个值。
- `t.scatter_(dim, index, value)`：按索引原地写值。`soft.scatter_(1, targets.unsqueeze(1), 1 - eps)` 把每行第 `targets[n]` 列设为 $1 - \varepsilon$；`index` 必须和 `t` 维数相同，所以要 `unsqueeze(1)` 成 `(N, 1)`。
- `torch.log_softmax(logits, dim=-1)` 或手写稳定版本。
- `.sum(dim=-1).mean()`。

## 常见坑

- **分母用 $C$**：本题非正确类别分到 $\varepsilon / (C - 1)$。注意 PyTorch 内置的 `F.cross_entropy(label_smoothing=ε)` 用的是另一种约定：所有类别（包括正确类别）都加 $\varepsilon / C$，正确类别为 $1 - \varepsilon + \varepsilon / C$，两者数值不同。
- **`scatter_` 的 index 维数不对**：`targets` 是一维的，要先 `unsqueeze(1)`。
- **不稳定的 $\log(\operatorname{softmax})$**：大 logits 时得到 $-\infty$，乘以 $q_c$ 后变成 NaN。
- 先对类别求和、再对样本求均值。
