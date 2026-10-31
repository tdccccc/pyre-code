## 背景

「GRPO 损失」一题是简化版：输入是整条回答一个对数概率，只做组内归一化，没有裁剪也没有 KL。本题把它补全成 DeepSeekMath 论文里的写法，对应 TRL 的 `loss_type="grpo"` 和 verl 的 `loss_agg_mode="seq-mean-token-mean"`。注意这已经不是两个框架的默认配置：它们的默认聚合都换成了 token 平均的变体，TRL 还默认 `beta=0`（不加 KL）。原始形式包括：

- 组内归一化得到每个回答一个优势 $A_i$，广播到这个回答的所有 token。
- 每个 token 套用 PPO 的裁剪比率，让同一批数据可以更新多步。
- 用 $k_3$ 估计器算出对参考模型的 KL，直接作为惩罚项写进损失。
- 先在每个回答内部按 token 平均，再对回答平均。

## 核心公式

**优势**：第 $i$ 个回答属于组 $g$，

$$
A_i = \frac{r_i - \operatorname{mean}(r_g)}{\operatorname{std}(r_g) + \varepsilon}
$$

标准差用**无偏估计**（除以 $G-1$），这是 `torch.std` 的默认行为，TRL 和 verl 都这样做。$\varepsilon$ 只防止除以零，取值各家不同（TRL 用 $10^{-4}$）。

**逐 token 损失**：

$$
\ell_{i,t} = -\Bigl[\min\bigl(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i\bigr) - \beta\, \text{KL}_{i,t}\Bigr],
\qquad \rho_{i,t} = \frac{\pi_\theta(o_{i,t})}{\pi_{\text{old}}(o_{i,t})}
$$

**KL 项**用 $k_3$ 估计器，记 $d = \log\pi_{\text{ref}} - \log\pi_\theta$：

$$
\text{KL}_{i,t} = e^{d} - d - 1 \ \ge 0
$$

**聚合**：

$$
L = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \ell_{i,t}
$$

（batch 里有多个提示时，对所有 $B$ 个回答取平均。）$|o_i|$ 是回答 $i$ 的有效 token 数。

## 形状

- `logps`、`old_logps`、`ref_logps`、`mask` 都是 `(B, T)`，`rewards` 是 `(B,)`。
- $B = P \cdot G$：$P$ 个提示，每个 $G$ 个回答，同组的回答在连续的行上，所以 `rewards.view(P, G)` 就能按组计算。
- 优势算出来是 `(B,)`，`view(-1, 1)` 后就能和 `(B, T)` 的 token 张量相乘。

## 用到的 PyTorch 操作

- `t.view(-1, G)`、`t.mean(dim=1, keepdim=True)`、`t.std(dim=1, keepdim=True)`：按组求统计量。
- `torch.minimum`、`torch.clamp`、`torch.exp`。
- `(x * m).sum(dim=1) / m.sum(dim=1)`：每行的掩码平均。

## 常见坑

- **用了总体标准差**：上一题规定 `unbiased=False`，本题跟随 TRL / verl 用无偏标准差。$G$ 小的时候两者差别明显（$G=2$ 时差 $\sqrt{2}$ 倍）。
- **整个 batch 一起做 token 平均**：这是 DAPO 提出、现在 TRL 和 verl 默认采用的聚合，但不是本题要求的 DeepSeekMath 原始形式。原始 GRPO 先在回答内平均，所以短回答的每个 token 权重更大。
- **KL 用了 $k_1$**：$\log\pi_\theta - \log\pi_{\text{ref}}$ 单个样本可以是负数，方差也更大。
- **KL 的符号写反**：损失是「负的目标」，KL 在目标里是减号，进到损失里就成了加号。写反了等于奖励策略偏离参考模型。
- **在整个 batch 上做归一化**：要按组。不同题目的难度不同，组均值才是那道题的基准。

## 延伸

- **比率在什么时候不等于 1**：如果每批数据只更新一步（TRL 默认 `num_iterations=1`），$\pi_\theta = \pi_{\text{old}}$，$\rho \equiv 1$，裁剪不起作用，梯度就是 $-A_i\nabla\log\pi$ 的平均。只有一批数据更新多步时裁剪才生效。
- **KL 为什么放在损失里**：PPO 把 KL 当成每个 token 的奖励，靠 critic 和 GAE 传播。GRPO 没有 critic，优势只来自最终的奖励，就直接把 KL 写进损失，不影响优势的计算。
- **$k_3$ 的梯度**：$k_3$ 的期望是 KL，但把它当损失求导，得到的梯度并不等于 KL 的梯度。2025 年有不少工作讨论这一点，有的实现改用 $k_1$ 或 $k_2$ 作为损失项。
- **越来越多的实现去掉 KL**：在可验证奖励（数学、代码）上，DAPO、Dr. GRPO 等工作把 $\beta$ 设为 $0$，TRL 的默认值也改成了 $0$。这样省掉参考模型的前向，策略也能走得更远。
- **长度偏差**：按回答内平均意味着，同样的优势，长回答每个 token 分到的权重更小。答错的长回答惩罚被稀释，模型会倾向于越写越长。Dr. GRPO 和 DAPO 都针对这一点改了聚合方式，框架的默认值也随之从按回答平均换成了 token 平均，见「损失聚合方式」一题。
- **框架里怎么配出本题的形式**：verl 用 `algorithm.adv_estimator=grpo`、`actor.use_kl_loss=True`、`actor.kl_loss_type=low_var_kl`（即 $k_3$）、`actor.loss_agg_mode=seq-mean-token-mean`；TRL 用 `loss_type="grpo"` 并把 `beta` 设为非零。

## 面试追问

1. **GRPO 和 PPO 的损失有什么区别？**
   裁剪的形式完全相同。区别在优势和 KL：GRPO 的优势来自组内归一化的奖励，同一回答的所有 token 共享；PPO 的优势来自 critic 和 GAE，每个 token 不同。GRPO 的 KL 写在损失里，PPO 的 KL 写在奖励里。
2. **为什么 KL 用 $k_3$ 而不是直接用对数比值？**
   采样自 $\pi_\theta$ 时，$k_1 = \log\pi_\theta - \log\pi_{\text{ref}}$ 无偏但单个样本可能是负数，方差大。$k_3$ 同样无偏，而且恒为非负，方差小得多。
3. **每批只更新一步时，裁剪还有用吗？**
   没有用。$\rho \equiv 1$，永远落在裁剪范围内。裁剪只在同一批数据更新多步、$\pi_\theta$ 偏离 $\pi_{\text{old}}$ 后才起作用。
4. **为什么现在很多 GRPO 训练把 KL 系数设成 0？**
   可验证奖励不太会被「刷分」，KL 约束的主要作用（防止奖励模型被钻空子）就不那么必要了。去掉以后省一次参考模型前向，模型的推理能力也能离初始模型更远。代价是失去对输出分布漂移的约束，需要靠熵等指标监控。
5. **组内标准差为 0 时会发生什么？**
   全对或全错的组，分子分母都是 0，$\varepsilon$ 保证结果是 0。这组回答没有策略梯度，只剩 KL 项，相当于浪费了这些采样。DAPO 的动态采样会把它们过滤掉，再补采样。
