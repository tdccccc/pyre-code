## 背景

「PPO 损失」一题只有一个裁剪项，输入是整条回答一个比率。真实的 RLHF 训练代码（verl、OpenRLHF、TRL）要多做三件事：

1. **逐 token 计算**：语言模型的每个 token 是一个动作，比率、优势、价值都是 `(B, T)`，不同回答长度不同，要用掩码把 padding 去掉。
2. **同时训练 critic**：除了策略损失，还要算价值网络的回归损失。verl 和 OpenRLHF 里 actor 和 critic 是两个独立的模型，各自用自己的损失和优化器，算的正是本题的这两项；把它们加成 $\text{pg} + c_1 \cdot \text{vf}$ 是 actor 和 critic 共享主干时的写法（原始 PPO 论文、TRL / trlx）。
3. **输出监控指标**：`clipfrac` 和 `approx_kl` 是 PPO 训练面板上必看的两条曲线，判断每一步更新是否过猛。

一次 PPO 迭代的流程：用当前策略生成一批回答，记下 `old_logps` 和 `old_values`；用奖励和 GAE 算出 `advantages` 和 `returns`（见「RLHF 奖励与 GAE」一题）；再把这批数据切成 mini-batch，更新若干轮，每轮都调用本题的损失。

## 核心公式

记 $m_t$ 为掩码，「掩码 token 平均」为 $\operatorname{mean}_m(x) = \sum m_t x_t / \sum m_t$，对整个 batch 的所有 token 一起求。

**策略损失**：PPO 的目标取负，原来的 $\min$ 变成 $\max$：

$$
\text{pg}_t = \max\bigl(-A_t\rho_t,\ -A_t\operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\bigr),\qquad \rho_t = e^{\log\pi_\theta - \log\pi_{\text{old}}}
$$

**价值损失**：critic 回归到 $R_t$（GAE 给出的回报），同样限制它一步不能走太远：

$$
v^{\text{clip}}_t = V_{\text{old},t} + \operatorname{clip}(V_t - V_{\text{old},t},\ -c_v,\ c_v),\qquad
\text{vf}_t = \tfrac{1}{2}\max\bigl((V_t - R_t)^2,\ (v^{\text{clip}}_t - R_t)^2\bigr)
$$

取 $\max$ 和策略损失的道理一样：用更悲观的那个误差，裁剪只会削弱更新、不会让损失看起来更好。

**总损失**：$L = \operatorname{mean}_m(\text{pg}) + c_1 \operatorname{mean}_m(\text{vf})$，$c_1$ 是 `vf_coef`。actor 和 critic 分开训练时没有 $c_1$，两项各自反向传播。

**监控指标**：

- $\text{clipfrac} = \operatorname{mean}_m\bigl(\mathbb{1}[-A\operatorname{clip}(\rho) > -A\rho]\bigr)$：裁剪分支真正生效的 token 比例。
- $\text{approx\_kl} = \operatorname{mean}_m(\log\pi_{\text{old}} - \log\pi_\theta)$：新旧策略之间 KL 的 $k_1$ 估计（见「KL 估计器」一题）。

## 形状

所有输入都是 `(B, T)`，`mask` 可以是 float 或 bool。输出是一个字典，五个值都是标量。

## 用到的 PyTorch 操作

- `torch.maximum(a, b)`：逐元素取较大值（`torch.max(a)` 是整体最大值，别混用）。
- `torch.clamp(x, lo, hi)`。
- `mask.to(logps.dtype)`：bool 掩码转成浮点数，才能相乘求和。
- `t.detach()` 或 `with torch.no_grad():`：常数输入和监控指标都不需要梯度。

## 常见坑

- **价值损失用了 $\min$**：critic 可以绕开裁剪一步跳到目标上，裁剪形同虚设。
- **clipfrac 只看 $\rho$ 是否越界**：$\rho > 1+\epsilon$ 但 $A<0$ 时，未裁剪项更大，用的是未裁剪分支，这个 token 并没有被裁剪。必须比较两个分支的损失。
- **先按序列平均再平均**：本题要求整个 batch 的 token 平均，短回答和长回答的每个 token 权重相同。两种做法的差别见「损失聚合方式」一题。
- **价值裁剪的中心用错**：裁剪围绕采样时的 $V_{\text{old}}$，不是围绕回报 $R$。
- **忘了 detach 优势**：优势由 critic 算出，不 detach 的话策略损失的梯度会流进价值网络。

## 延伸

- **熵奖励**：经典 PPO 还会减去 $c_2 \cdot \operatorname{mean}_m(\mathcal{H}[\pi])$ 鼓励探索。LLM 的词表很大，算熵要完整的 logits，RLHF 里通常把系数设成 $0$，靠 KL 惩罚约束策略。
- **dual-clip**：$A<0$ 时，$\rho$ 很大会让 $-A\rho$ 无上限地增大。dual-clip 再加一个上限 $-A \cdot c$。本题没有加，但 verl 默认开启，$c=3$，对应配置 `clip_ratio_c`。
- **优势白化**：常见做法是先在整个 batch 上把优势标准化到均值 0、方差 1，再算损失，避免奖励尺度变化导致学习率实际上忽大忽小。
- **价值裁剪有争议**：Engstrom 等人的 *Implementation Matters in Deep Policy Gradients* 把它列为 PPO 里的一项代码级技巧；Andrychowicz 等人的 *What Matters in On-Policy Reinforcement Learning?* 在大规模对比中发现价值裁剪反而会损害效果。所以很多实现提供开关。
- **监控指标怎么读**：
  - `clipfrac` 的正常范围因任务而异，经典 PPO 里常在 $0.1$ 上下，LLM 上通常更小；持续上升说明学习率太大或每批更新轮数太多。
  - `approx_kl` 突然变大也是同一个信号。有的实现在它超过阈值时提前停止这一批的更新（early stop）。
- **critic 的初始化**：critic 常从奖励模型初始化，最后一层换成输出标量的价值头。它和策略一样大，这是 PPO 显存开销大的主要原因。

## 面试追问

1. **为什么价值损失也要裁剪？**
   和策略裁剪的目的一样：同一批数据要更新多轮，限制价值函数每次相对采样时的估计不要偏离太远，防止 critic 过拟合当前这批回报。用 $\max$ 取悲观的误差，保证裁剪只会让更新更保守。
2. **clipfrac 和 approx_kl 怎么用来调参？**
   两者都衡量新策略离采样策略有多远。持续偏高说明步子太大，可以降学习率、减少每批的更新轮数或调小 $\epsilon$；接近 $0$ 说明几乎没学到东西，可以适当放大。
3. **为什么 token 级的损失要按整个 batch 的 token 数平均？**
   这样每个 token 的权重相同，长回答提供的梯度更多。按序列平均会让短回答的 token 权重更大，带来长度偏差。不同框架的默认做法不同，verl 用 `loss_agg_mode` 选择。
4. **PPO 一次迭代要跑哪些前向？**
   生成阶段策略模型自回归采样；之后参考模型、奖励模型、critic 各做一次前向，得到 `ref_logps`、分数和 `old_values`；训练阶段策略和 critic 每个 mini-batch 再做前向和反向。四个模型的显存加上生成阶段的 KV cache，是 PPO 工程上最重的地方。
5. **为什么把 KL 放进奖励而不是损失？**
   经典 RLHF（InstructGPT）把 $-\beta\,\text{KL}$ 当作每个 token 的奖励，通过 GAE 和优势一起传播到前面的 token，critic 会把它也学进价值里。GRPO 没有 critic，就直接把 KL 写进损失。两种写法都能约束策略，只是作用路径不同。
