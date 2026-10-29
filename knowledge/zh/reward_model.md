## 背景

RLHF（基于人类反馈的强化学习）的经典流程分三步：

1. **SFT**：用人工写的示范回答做监督微调
2. **训练奖励模型**：让人对同一个问题的两个回答做比较，选出更好的那个；训练一个模型给回答打分，使「被选中」的回答得分更高
3. **强化学习**：用奖励模型的分数作为奖励，用 PPO 等算法优化语言模型

人类很难给出绝对分数（这个回答值 7.3 分？），但比较两个回答哪个更好则容易得多。所以奖励模型从**成对比较**中学习。

## 核心公式

**奖励**：取回答最后一个 token 的隐藏状态 $h$，用一个线性头投影成标量：$r = h^{\top} w$。

**Bradley-Terry 模型**：假设人选择回答 $y_w$ 而非 $y_l$ 的概率由两者奖励之差决定：

$$
P(y_w \succ y_l) = \frac{e^{r_w}}{e^{r_w} + e^{r_l}} = \sigma(r_w - r_l),
\qquad
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**损失**：最大化观测到的偏好的对数似然，即最小化

$$
L = -\frac{1}{B} \sum_{n=1}^{B} \log \sigma\bigl(r_w^{(n)} - r_l^{(n)}\bigr)
$$

**几个性质**：

- 只有差值 $\Delta = r_w - r_l$ 有意义，所有奖励加同一个常数不影响损失。
- $\Delta = 0$（分不出好坏）时 $L = -\log \tfrac12 = \log 2 \approx 0.693$。
- $\Delta \to +\infty$ 时 $L \to 0$；$\Delta$ 越大损失越小。
- 梯度：$\dfrac{\partial L}{\partial \Delta} = \sigma(\Delta) - 1$，当模型已经很确定（$\sigma(\Delta) \approx 1$）时梯度很小。

**数值稳定的写法**：$\log \sigma(x) = -\log(1 + e^{-x})$。$x$ 很负时 $e^{-x}$ 会溢出，实际工程中用 `F.logsigmoid` 或 $\operatorname{softplus}$ 计算。本题要求手写 sigmoid。

## 形状

| 量 | 形状 |
| --- | --- |
| `chosen_hidden`、`rejected_hidden` | `(B, D)` |
| `reward_head` | `(D, 1)` |
| 奖励 $r_w$、$r_l$ | `(B, 1)`，可 `squeeze(-1)` 成 `(B,)` |
| 输出 | 标量 |

## 用到的 PyTorch 操作

- `@` 矩阵乘、`t.squeeze(-1)`。
- `torch.exp`、`torch.log`、`.mean()`。

## 常见坑

- **差值方向反了**：是 $r_{\text{chosen}} - r_{\text{rejected}}$。
- **使用库函数**：本题要求手写 $\sigma(x) = 1/(1 + e^{-x})$，不要调用 `torch.sigmoid`。
- **漏了负号**：损失是 $-\log \sigma(\cdot)$，为正数。
- 用 `mean` 而不是 `sum`。
