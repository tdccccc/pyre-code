## 背景

PPO 需要一个**价值网络**（critic）来估计「这个状态平均能拿多少奖励」，作为比较的基准。对大模型来说，价值网络通常和策略模型一样大，显存和计算都翻倍。

**GRPO**（Group Relative Policy Optimization，DeepSeekMath / DeepSeek-R1 使用）去掉了价值网络：对同一个问题采样一组（$G$ 个）回答，用**组内平均奖励**作为基准。比平均好的回答被鼓励，比平均差的被抑制。

## 核心公式

**策略梯度**：要最大化期望奖励 $J = \mathbb{E}_{y \sim \pi}[r(y)]$，其梯度为

$$
\nabla J = \mathbb{E}_{y \sim \pi}\bigl[\, r(y)\, \nabla \log \pi(y) \,\bigr]
$$

**基准（baseline）**：把 $r$ 换成 $r - b$（$b$ 与 $y$ 无关），梯度的期望不变，因为

$$
\mathbb{E}_{y \sim \pi}\bigl[\, b\, \nabla \log \pi(y) \,\bigr] = b \sum_y \pi(y) \frac{\nabla \pi(y)}{\pi(y)} = b\, \nabla \sum_y \pi(y) = b\, \nabla 1 = 0
$$

但方差会大大降低：奖励全是正数时，不减基准的话所有回答都会被「鼓励」，只是程度不同。

**组内归一化的优势**：同一组（同一个问题）的回答 $i$，

$$
A_i = \frac{r_i - \operatorname{mean}_g(r)}{\operatorname{std}_g(r) + \epsilon}
$$

除以标准差让不同难度的问题在同一尺度上：有的问题奖励差异大，有的差异小。

**损失**（本题的简化版，不含裁剪和 KL 项）：

$$
L = -\frac{1}{B} \sum_{i=1}^{B} A_i \cdot \log \pi(y_i)
$$

$A_i$ 被当作常数（detach），梯度只流经 $\log \pi$。

## 形状

`logps`、`rewards`、`group_ids`、优势值都是 `(B,)`，输出是标量。同一个 `group_id` 的元素在 batch 中不一定相邻。

## 用到的 PyTorch 操作

- `group_ids.unique()`：所有不同的组号。
- 布尔掩码 `mask = group_ids == gid`，`rewards[mask]` 取出一组，`adv[mask] = ...` 写回。
- `t.std(unbiased=False)`：总体标准差（除以 $n$）。
- `t.detach()`。

## 常见坑

- **在整个 batch 上归一化**：要按组分别减去组均值、除以组标准差。
- **用了无偏标准差**：本题用总体标准差 `unbiased=False`。
- **没有 detach 优势值**：梯度会流进奖励的归一化计算。
- **只有一个元素的组**：标准差为 $0$，$\epsilon$ 防止除以零，此时优势为 $0$。
- 损失前面有负号：最小化损失 = 最大化 $A \cdot \log \pi$。
