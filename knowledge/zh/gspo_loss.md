## 背景

GRPO 的奖励和优势属于整条回答，但重要性比率 $\rho_{i,t} = \pi_\theta(o_{i,t}) / \pi_{\text{old}}(o_{i,t})$ 却是逐 token 算的。重要性采样要靠多个样本平均才能修正分布差异，而每个位置只有一个采样到的 token，这个比率更像是噪声，回答越长噪声累积越多。Qwen 团队在训练 Qwen3（尤其是 MoE 模型）时发现这会导致训练崩溃，于是提出了 **GSPO**（Group Sequence Policy Optimization，2025）：

- 比率按**整条回答**算，和奖励的粒度一致；
- 用长度归一化，让长短不同的回答比率在同一个尺度上；
- 裁剪也以回答为单位，一条回答要么整体参与更新，要么整体被裁掉。

优势仍然用 GRPO 的组内归一化（见「组内基线：GRPO / Dr. GRPO / RLOO / REINFORCE++」），本题直接作为输入给出。逐 token 比率的写法见「完整版 GRPO 损失」。

## 核心公式

**序列比率**：回答 $i$ 的有效长度为 $|o_i|$，

$$
s_i = \Bigl(\frac{\pi_\theta(o_i)}{\pi_{\text{old}}(o_i)}\Bigr)^{1/|o_i|} = \exp\Bigl(\frac{1}{|o_i|}\sum_{t} \log\frac{\pi_\theta(o_{i,t})}{\pi_{\text{old}}(o_{i,t})}\Bigr)
$$

也就是逐 token 比率的**几何平均**。

**目标**（裁剪区间不对称）：

$$
L = -\frac{1}{B}\sum_i \min\bigl(s_i A_i,\ \operatorname{clip}(s_i,\ 1-\epsilon_{\text{low}},\ 1+\epsilon_{\text{high}})\, A_i\bigr)
$$

论文里 $\epsilon_{\text{low}} = 3\times10^{-4}$、$\epsilon_{\text{high}} = 4\times10^{-4}$。

**梯度**：在裁剪范围内，

$$
\nabla_\theta L_i \propto -A_i\, s_i \cdot \frac{1}{|o_i|}\sum_t \nabla_\theta \log \pi_\theta(o_{i,t})
$$

同一回答的所有 token 权重相同。对比 GRPO：每个 token 的梯度乘的是它自己的 $\rho_{i,t}$，权重参差不齐。

## 形状

- `logps`、`old_logps`、`mask` 都是 `(B, T)`，`advantages` 是 `(B,)`。
- 先在 `dim=1` 上做掩码求和、除以长度，得到 `(B,)` 的对数比率，取 `exp` 就是 $s_i$。
- 之后的裁剪和取 $\min$ 都在 `(B,)` 上做，最后对 $B$ 个回答取平均。

## 用到的 PyTorch 操作

- `(x * m).sum(dim=1) / m.sum(dim=1)`：每行的掩码平均。
- `torch.exp`、`torch.clamp(t, lo, hi)`、`torch.minimum`。
- `t.detach()`：`old_logps` 和优势是常数。

## 常见坑

- **对 token 比率取算术平均**：$\frac{1}{|o|}\sum_t \rho_t$ 不等于 $\exp\bigl(\frac{1}{|o|}\sum_t \log\rho_t\bigr)$。要在对数空间平均再取指数。
- **忘了除以长度**：直接用整条序列的似然比 $\pi_\theta(o)/\pi_{\text{old}}(o)$，回答越长比率偏离 1 越远，长回答几乎都会被裁掉，短回答几乎不会。
- **按 token 裁剪**：那就退回了 GRPO。GSPO 的裁剪作用在 $s_i$ 上，一条回答的所有 token 一起被保留或丢弃。
- **两侧用了同一个裁剪幅度**：`clip_low` 和 `clip_high` 是两个参数。
- **没有 detach `old_logps`**：梯度会流进生成数据的旧策略。

## 延伸

- **为什么裁剪范围这么小**：$s_i$ 是几百上千个 token 比率的几何平均，各 token 的偏差在平均中相互抵消，$s_i$ 本来就离 1 很近。$3\times10^{-4}$ 这个量级对应的是平均每个 token 的对数比率的偏差。论文报告 GSPO 裁掉的 token 比例比 GRPO 高出约两个数量级，训练效率反而更高，说明 GRPO 的逐 token 梯度里有大量噪声。
- **MoE 的路由问题**：MoE 模型每次更新后，同一个 token 激活的专家可能变了，逐 token 的比率因此剧烈波动。GRPO 训练 MoE 时需要路由回放（Routing Replay，更新时强制使用采样时的专家）才能稳定；GSPO 只看整条回答的似然，对单个 token 的波动不敏感，不需要这个技巧。
- **GSPO-token**：论文还给了一个变体，让每个 token 可以有不同的优势：$s_{i,t} = \operatorname{sg}[s_i] \cdot \pi_\theta(o_{i,t}) / \operatorname{sg}[\pi_\theta(o_{i,t})]$，$\operatorname{sg}$ 表示停止梯度。数值上等于 $s_i$，当所有 token 优势相同时与 GSPO 完全一样，用于多轮或需要 token 级奖励的场景。
- **训练和推理的精度差异**：论文认为序列级的比率对推理引擎和训练框架之间的数值误差更不敏感，有望直接使用推理引擎给出的对数概率，省掉一次训练侧的重新计算。
- **框架**：TRL 的 `GRPOConfig(importance_sampling_level="sequence")`、verl 的 `actor.policy_loss.loss_mode=gspo` 都实现了这种序列级比率。

## 面试追问

1. **GSPO 和 GRPO 的核心区别？**
   优势一样，都是组内归一化的奖励。区别在重要性比率：GRPO 每个 token 一个比率、逐 token 裁剪；GSPO 每条回答一个按长度归一化的比率，整条回答一起裁剪。
2. **为什么 GRPO 的逐 token 比率有问题？**
   重要性采样需要对同一分布的多个样本加权平均才有意义，而每个 token 位置只有一个样本，比率只是噪声。奖励又是序列级的，用 token 级的修正与目标的粒度不匹配。噪声随长度累积，在长回答和 MoE 上容易让训练崩溃。
3. **为什么序列比率要开 $|o_i|$ 次方？**
   不归一化的话，整条序列的似然比是几百个 token 比率的连乘，长回答的比率会极大或极小，几乎全部被裁掉，裁剪范围也没法对不同长度统一设置。开 $|o_i|$ 次方后，各长度的比率都在 1 附近，同一个裁剪范围适用于所有回答。
4. **GSPO 的裁剪比例比 GRPO 高得多，为什么效率反而更好？**
   被裁掉的是整条偏离过大的回答，留下的回答梯度干净：每个 token 权重相同，不受单个 token 比率噪声的影响。GRPO 虽然保留了更多 token，但很多梯度是噪声。
5. **GSPO 还需要 KL 项吗？**
   GSPO 改的是比率和裁剪，KL 项可加可不加，和 GRPO 一样写进损失即可。在可验证奖励上，和 DAPO 一样常见的做法是去掉 KL，靠裁剪限制每步更新的幅度。
