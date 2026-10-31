## 背景

奖励模型（或判题器）给**整条回答**打一个分，但 PPO 的动作是**每个 token**。中间要补一步：把序列级分数分配到每个 token 上，再估计每个 token 的**优势**（这一步比平均水平好多少），以及 critic 要拟合的**回报**。

InstructGPT 以来的标准做法（OpenRLHF、verl、TRL 的 PPO 都一样）：

1. 每个回答 token 扣一份相对参考模型的 KL 惩罚，让策略别跑太远；
2. 在**最后一个回答 token** 上加上序列分数；
3. 用 critic 的价值 $V_t$，从后往前做广义优势估计（GAE）。

## 核心公式

**逐 token 奖励**（一行回答长度为 $L$，$\beta$ 是 KL 系数）：

$$
r_t = -\beta \cdot \text{kl}_t + \mathbb{1}[t = L-1] \cdot \text{score}, \qquad t = 0, \dots, L-1
$$

**TD 误差**：

$$
\delta_t = r_t + \gamma V_{t+1} - V_t, \qquad V_L := 0
$$

最后一个 token 之后回答就结束了，是终止状态，没有未来价值。

**GAE** 把多步 TD 误差按 $(\gamma\lambda)$ 指数加权：

$$
\hat{A}_t = \sum_{l \ge 0} (\gamma \lambda)^l \, \delta_{t+l} = \delta_t + \gamma\lambda\, \hat{A}_{t+1}
$$

**回报**（critic 的回归目标）：

$$
R_t = \hat{A}_t + V_t
$$

- $\gamma$：折扣因子。LLM 的回答很短、奖励只在结尾，RLHF 里通常取 $1$。
- $\lambda$：在偏差和方差之间折中。$\lambda = 0$ 时 $\hat{A}_t = \delta_t$，只看一步，依赖 critic，方差小、偏差大；$\lambda = 1$ 时 $\hat{A}_t = \sum_l \gamma^l r_{t+l} - V_t$，就是蒙特卡洛回报减基线，无偏但方差大。

## 形状

`scores` 是 `(B,)`，`kl`、`values`、`mask` 都是 `(B, T)`，两个输出都是 `(B, T)`。每行的回答从第 0 列开始、右侧填充，所以最后一个回答 token 的列号是 `mask[b].sum() - 1`。

## 用到的 PyTorch 操作

- `mask.sum(dim=1)`：每行的回答长度；配合高级索引 `x[torch.arange(B), idx]` 给每行的某一列加值。
- 切片错位：`values[:, 1:]` 对齐到 `t` 位置就是 $V_{t+1}$。
- 倒序循环 `for t in reversed(range(T))`：GAE 是从后往前的递推，时间维上没法直接向量化（$T$ 不大，循环就够）。
- `torch.no_grad()`：优势和回报是目标值，不参与反向传播。

## 常见坑

- **分数加在最后一列**：右侧有填充的行，最后一列是填充，分数就丢了。要加在每行自己的最后一个回答 token 上。
- **从填充位置自举**：最后一个回答 token 的 $V_{t+1}$ 必须是 $0$，不能读下一列（那是填充，值是垃圾）。
- **递推里漏了 $\lambda$**：写成 $\hat{A}_t = \delta_t + \gamma \hat{A}_{t+1}$，就退化成 $\lambda=1$ 的蒙特卡洛优势。
- **KL 符号反了**：KL 是惩罚，奖励里要减。
- **回报算成奖励累加**：critic 的目标是 $\hat{A}_t + V_t$（即 TD($\lambda$) 回报），不是单纯的奖励累加，两者只在 $\lambda = 1$ 时相同。

## 延伸

- **KL 用哪个估计器**：奖励里的 KL 通常用 k1，即 $\log \pi_{\text{old}} - \log \pi_{\text{ref}}$（采样时的策略），见「KL 估计器」一题。这里 KL 进了奖励，再经 GAE 传给前面所有 token。
- **优势标准化**：GAE 算完后，常对整个 batch 的有效 token 做一次 whitening（减均值、除标准差），让不同 batch 的梯度尺度稳定。本题不做，留给训练循环。
- **为什么 critic 难训**：奖励只在最后一个 token 出现，前面 token 的价值全靠 critic 从稀疏信号里学，而且 critic 通常和策略一样大。这正是 GRPO 用组内平均代替 critic 的动机。
- **critic 的初始化**：一般从奖励模型初始化，并在正式训练前先单独训几步 critic（warmup），否则一开始的优势全是噪声。
- **代价**：计算本身很便宜，是 $O(BT)$ 的循环；真正贵的是得到 `values` 所需的 critic 前向，以及维护整个 critic 模型。

## 面试追问

1. **为什么分数只加在最后一个 token 上？**
   奖励模型只对完整回答打分，中间 token 没有单独的奖励信号；由 GAE 和 critic 把这个延迟奖励「归功」给前面的 token。
2. **$\lambda$ 取 0 和取 1 分别是什么？**
   $0$：一步 TD 误差，完全依赖 critic，方差小、偏差大。$1$：蒙特卡洛回报减基线，无偏、方差大。常用 $0.95$ 左右折中。
3. **RLHF 里 $\gamma$ 为什么常取 1？**
   回答长度有限，奖励只在结尾；折扣会让前面 token 的优势系统性偏小，没有必要。
4. **KL 惩罚放在奖励里和放在损失里有什么区别？**
   放在奖励里，KL 通过优势影响策略梯度，也会被 critic 学进价值；放在损失里（GRPO）是一个直接的正则项，不经过优势。
5. **最后一个 token 为什么不能用下一个位置的价值自举？**
   回答已经结束，没有下一个状态；下一列是填充，它的值没有意义，读进来会污染所有前面 token 的优势。
