## 背景

策略梯度每采样一批数据只能更新一步：更新后策略变了，旧数据就不再是新策略采出来的。采样很贵（对 LLM 来说要生成完整回答），一批数据只用一次太浪费。

**PPO**（Proximal Policy Optimization）用**重要性采样**修正分布差异，让同一批数据可以更新多步；同时**裁剪**更新幅度，防止新策略离采样时的旧策略太远而导致训练崩溃。它是 RLHF 中最经典的算法。

## 核心公式

**重要性采样比率**：

$$
\rho = \frac{\pi_{\text{new}}(a)}{\pi_{\text{old}}(a)} = \exp\bigl(\log \pi_{\text{new}}(a) - \log \pi_{\text{old}}(a)\bigr)
$$

用对数概率之差再取指数计算，比直接相除数值上更稳定。

**代理目标**：$\mathbb{E}_{a \sim \pi_{\text{old}}}[\rho\, A]$ 的梯度在 $\pi_{\text{new}} = \pi_{\text{old}}$ 处等于策略梯度。

**裁剪目标**（$\epsilon$ 通常取 $0.2$）：

$$
L^{\text{CLIP}} = \mathbb{E}\Bigl[\min\bigl(\rho A,\ \operatorname{clip}(\rho,\ 1 - \epsilon,\ 1 + \epsilon)\, A\bigr)\Bigr]
$$

损失取负：$L = -L^{\text{CLIP}}$。

**分情况看**：

| 优势 | 含义 | 目标随 $\rho$ 的变化 |
| --- | --- | --- |
| $A > 0$ | 这个动作比平均好，要提高它的概率 | $\rho$ 增大到 $1 + \epsilon$ 后不再增加 |
| $A < 0$ | 这个动作比平均差，要降低它的概率 | $\rho$ 减小到 $1 - \epsilon$ 后不再增加 |

一旦超出裁剪范围，目标变成常数，梯度为 $0$，策略就不会被继续往那个方向推。取 $\min$ 意味着总是采用**更保守**（更悲观）的那个估计：裁剪只能削弱改进的幅度，不会因为裁剪反而让目标看起来更好。

## 形状

`new_logps`、`old_logps`、`advantages` 都是 `(B,)`，输出是标量。

## 用到的 PyTorch 操作

- `torch.exp`、`torch.clamp(t, lo, hi)`。
- `torch.min(a, b)`：两个张量逐元素取较小值（不是求整体最小值）。
- `t.detach()`：旧策略的对数概率和优势是常数。

## 常见坑

- **用 `max` 代替 `min`**：变成了乐观估计，裁剪失去作用。
- **裁剪了优势而不是比率**：$\operatorname{clip}$ 作用在 $\rho$ 上。
- **只用裁剪项**：必须和未裁剪项取 $\min$。比如 $A < 0$ 且 $\rho > 1 + \epsilon$ 时，未裁剪项更小，必须保留它的梯度，才能把这个坏动作的概率拉回来。
- **没有 detach**：梯度只应流经 `new_logps`。
- 损失要取负号。
