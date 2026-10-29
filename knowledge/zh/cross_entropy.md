## 背景

分类模型对每个类别输出一个分数（logit），softmax 把它们变成概率。训练目标是让**正确类别的概率尽量大**。交叉熵损失就是正确类别概率的负对数，语言模型的预训练损失（预测下一个 token）也正是它。

## 核心公式

logits $z \in \mathbb{R}^{C}$，正确类别 $y$：

$$
p_c = \frac{e^{z_c}}{\sum_{k} e^{z_k}},
\qquad
\ell = -\log p_y
$$

$p_y \to 1$ 时损失 $\to 0$；$p_y \to 0$ 时损失 $\to \infty$。

展开对数，得到直接从 logits 计算的形式：

$$
\ell = -z_y + \log \sum_{k} e^{z_k} = \operatorname{LSE}(z) - z_y
$$

**log-sum-exp 技巧**：$\operatorname{LSE}$ 直接算会因 $e^{z_k}$ 溢出得到 $\infty$。提出最大值 $m = \max_k z_k$：

$$
\operatorname{LSE}(z) = m + \log \sum_{k} e^{z_k - m}
$$

此时所有指数都 $\le 0$，求和至少为 $1$，既不会上溢，也不会出现 $\log 0$。

**梯度**非常简洁：

$$
\frac{\partial \ell}{\partial z_c} = p_c - \mathbb{1}[c = y]
$$

即「预测概率 − one-hot 标签」。

**batch 平均**：$B$ 个样本的损失取均值 $L = \frac{1}{B} \sum_{n=1}^{B} \ell_n$。

## 形状

| 量 | 形状 |
| --- | --- |
| `logits` | `(B, C)` |
| `targets` | `(B,)`，整数类别 |
| 每个样本的 $\log p$ | `(B, C)` |
| 取出 $\log p_{n, y_n}$ | `(B,)` |
| 输出 | 标量 |

## 用到的 PyTorch 操作

- `torch.logsumexp(z, dim=-1, keepdim=True)`：内置的稳定 LSE。
- **成对索引**：`t[torch.arange(B), targets]` 取出每一行 $n$ 的第 `targets[n]` 列，得到 `(B,)`。
- `.mean()`。

## 常见坑

- **先 softmax 再 `log`**：当 $p_y$ 下溢为 $0$ 时得到 $-\infty$，损失变成 $\infty$ 或 NaN。测试会用很大的 logits 检查。
- **`t[:, targets]`**：这会取出所有行的所有目标列，得到 `(B, B)`，不是每行一个值。
- **用 `sum` 代替 `mean`**：要和 `F.cross_entropy` 的默认行为（取均值）一致。
- **符号**：损失是 $-\log p_y$，为正数。
