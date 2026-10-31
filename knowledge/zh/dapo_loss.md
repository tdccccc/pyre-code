## 背景

**DAPO**（Decoupled Clip and Dynamic sAmpling Policy Optimization，字节 Seed 与清华 AIR，2025）是在 GRPO 基础上为**长思维链 RL** 做的一组修补。论文发现直接用 GRPO 训练 Qwen2.5-32B 基础模型会出现熵塌缩、奖励噪声大、训练不稳定等问题，于是提出四项改动，在 AIME 2024 上用约一半的训练步数超过了 DeepSeek-R1-Zero-Qwen-32B：

1. **Clip-Higher**：把裁剪的上下界解耦，只放宽上界，防止熵塌缩。
2. **Dynamic Sampling**：丢掉全对或全错的组，保证每个 batch 都有有效梯度。
3. **Token-Level Policy Gradient Loss**：在所有 token 上平均，而不是先在回答内平均。
4. **Overlong Reward Shaping**：对超出长度预算的回答做平滑的长度惩罚。

此外 DAPO **去掉了 KL 项**。论文给出的理由是：训练长思维链推理模型时，策略的分布预期会和初始模型差别很大，KL 约束没有必要。

## 核心公式

**超长惩罚**：记 $S = L_{\max} - L_{\text{cache}}$，

$$
R_{\text{len}}(L) = \begin{cases} 0 & L \le S \\ \dfrac{S - L}{L_{\text{cache}}} & S < L \le L_{\max} \\ -1 & L > L_{\max} \end{cases}
$$

在 $(S, L_{\max}]$ 这一段从 $0$ 线性降到 $-1$。论文取 $L_{\max} = 20480$、$L_{\text{cache}} = 4096$，即长度在 16384 以内不罚。

**优势**：只在保留的组内，对加了惩罚的奖励 $\tilde r_i = r_i + R_{\text{len}}(L_i)$ 做组内标准化，和 GRPO 一样。

**目标**（取负即损失）：

$$
\mathcal{J} = \frac{1}{\sum_i |o_i|}\sum_{i}\sum_{t=1}^{|o_i|} \min\bigl(\rho_{i,t} A_i,\ \operatorname{clip}(\rho_{i,t},\ 1-\epsilon_{\text{low}},\ 1+\epsilon_{\text{high}})\, A_i\bigr)
$$

其中 $i$ 只取保留下来的组里的回答，论文取 $\epsilon_{\text{low}} = 0.2$、$\epsilon_{\text{high}} = 0.28$。

## 形状

- `logps`、`old_logps`、`mask` 是 `(B, T)`；`rewards`、`lengths` 是 `(B,)`。
- `rewards.view(-1, G)` 得到 `(P, G)`，按行判断每组是否保留，得到 `keep`，形状 `(P,)`。
- `keep.repeat_interleave(G)` 把组级的保留标记展开回 `(B,)`，再和 `mask` 相乘，被丢掉的行就整行变成 0。

## 用到的 PyTorch 操作

- `t.max(dim=1).values`、`t.min(dim=1).values`：每组的最大、最小奖励，两者相等就说明全组相同。
- `torch.where(cond, a, b)`：分段函数，嵌套两层就能写出三段的惩罚。
- `t.repeat_interleave(G)`：`[a, b]` 变成 `[a, a, …, b, b, …]`，把组标记扩展到每个回答。
- `t.std(dim=1, keepdim=True)`：默认无偏标准差。
- 全部被过滤时，要返回一个值为 0、但仍由 `logps` 计算得到的标量（比如让 `logps` 参与一次乘以零的运算再求和），这样 `backward()` 后梯度是全 0，而不是 `None`。

## 常见坑

- **用加了惩罚后的奖励做过滤**：全错的组里回答长度不同，加上长度惩罚后奖励就不再相同，会被误留下。动态采样看的是「答对与否」，必须在原始奖励上判断。
- **上下界用了同一个 $\epsilon$**：那就退化成普通 PPO 裁剪，`eps_high` 失去作用。
- **在回答内先平均**：DAPO 要求在所有保留的 token 上平均。
- **被丢掉的组只是把优势置零**：它们的 token 仍然计入分母，平均值就被稀释了。要整行从掩码里去掉。
- **惩罚的分段写错**：边界 $L = S$ 时惩罚为 0，$L = L_{\max}$ 时恰好为 $-1$，中间线性。斜率写反就会变成刚超过 $S$ 时罚得最重。
- **全部被过滤时返回 Python 的 `0.0` 或新建的张量**：训练循环里还要 `backward()`，返回值必须连在 `logps` 上。

## 延伸

- **为什么放宽上界能防止熵塌缩**：裁剪限制的是比率 $\rho$，对概率小的 token 影响最大。旧概率 $0.01$ 的 token，在 $\epsilon = 0.2$ 下一步最多涨到 $0.012$；旧概率 $0.9$ 的 token 却几乎不受限制。结果是高概率 token 越来越确定，探索性的低概率 token 很难被提上来，熵迅速下降。只放宽上界给低概率 token 更多上升空间；下界保持不变，否则会把一些 token 的概率压到接近 0，反而让采样空间塌缩。
- **动态采样的代价**：训练后期越来越多的题模型全答对，有效 batch 越来越小，梯度方差变大。DAPO 的做法是**过采样**：不断采样、过滤，直到凑满一个 batch 的有效组，代价是额外的生成开销。本题只做过滤这一步。
- **为什么是软惩罚**：被截断的回答直接给负奖励，会把「推理正确但写得长」也一起惩罚，给奖励引入噪声。论文先试了**超长过滤**（overlong filtering，被截断的样本不计损失），发现训练更稳，再提出这种平滑惩罚，让模型在长度预算附近逐渐收敛，而不是撞墙。
- **在框架里**：TRL 的 `GRPOConfig` 有 `epsilon_high`（clip-higher）、`loss_type="dapo"`（token 平均）、`mask_truncated_completions`（超长过滤）；verl 的 DAPO recipe 里有对应的 `clip_ratio_low` / `clip_ratio_high`、组过滤和超长缓冲区配置。
- **与其他改进的关系**：Dr. GRPO 进一步去掉了标准差归一化和长度归一化；GSPO 把比率和裁剪从 token 级挪到序列级。这些改动可以组合使用，见「组内基线：GRPO / Dr. GRPO / RLOO / REINFORCE++」和「GSPO 序列级比率损失」两题。

## 面试追问

1. **Clip-Higher 为什么只放宽上界，不放宽下界？**
   上界限制的是好 token 概率的上涨，对低概率的探索性 token 影响最大，放宽它能防止熵塌缩。下界限制的是坏 token 概率的下降，放宽它会让一些 token 的概率被压到接近 0，减少采样多样性，反而加速塌缩。
2. **动态采样解决什么问题？有什么代价？**
   全对或全错的组优势全为 0，不贡献梯度，只占 batch 的位置，有效样本数随训练进行越来越少，梯度噪声越来越大。过滤并补采样保证每个 batch 都是有效样本，代价是额外的生成开销。
3. **为什么超长回答用软惩罚，而不是直接判错？**
   被截断的回答可能推理是对的，只是没写完，直接判错会给奖励引入噪声，模型分不清是「错了」还是「太长」。软惩罚在长度预算附近逐渐加大惩罚，给出平滑的信号。
4. **DAPO 为什么去掉了 KL？**
   论文的理由是：长思维链推理的训练中，模型分布预期会和初始模型差别很大，KL 惩罚会拖住这种必要的变化，所以不再需要。
5. **token 级平均相比 GRPO 原始的聚合有什么好处？**
   GRPO 原始写法先在回答内平均，长回答的每个 token 权重更小，其中的重复、胡言乱语得不到足够惩罚，正确长回答里的好推理也得不到足够奖励。token 级平均让每个 token 权重相同。
