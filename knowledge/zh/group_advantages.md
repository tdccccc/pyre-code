## 背景

PPO 用 critic 估计每个 token 的价值，再用 GAE 得到优势。critic 和策略一样大，训练也不稳定。在大模型 RL 里，奖励通常只在回答结束时给一次，于是出现了一类**无 critic** 的做法：对同一个提示采样 $G$ 个回答，用**同组其他回答的奖励**当基线（baseline），算出每个回答一个优势，再广播到它的所有 token。

这一类方法的差别只在三点：基线怎么取，要不要除以标准差，在哪个范围上除。框架里它们是并列的选项：

- verl 的 `algorithm.adv_estimator`（`grpo`、`rloo`、`reinforce_plus_plus_baseline` 等），`norm_adv_by_std_in_grpo=False` 就是 Dr. GRPO。
- OpenRLHF 的 `advantage_estimator`（`group_norm`、`rloo`、`reinforce_baseline`、`dr_grpo` 等）。其中只有 `reinforce_baseline` 会在减完组均值后再做全局白化，`rloo`、`group_norm`、`dr_grpo` 不做。
- TRL 的 `scale_rewards` 决定优势除以哪个标准差：`"group"`（默认，组内标准差，即 GRPO）、`"batch"`（整个 batch 的标准差，和本题的 `reinforce_pp` 很接近）、`"none"`（不除，即 Dr. GRPO）。TRL 的 `loss_type` 只管损失怎么聚合，和优势无关。

优势算出来之后怎么用，见「完整版 GRPO 损失」；聚合方式见「损失聚合方式」。

## 核心公式

策略梯度里减去一个基线 $b$：

$$
\nabla J = \mathbb{E}\bigl[(r_i - b)\, \nabla \log \pi(o_i)\bigr]
$$

只要 $b$ 不依赖回答 $o_i$ 本身，减掉它不改变梯度的期望，只降低方差。$r_i$ 是回答 $i$ 的奖励，$\bar r_g$ 是它所在组的平均奖励，$G$ 是组大小。

| 方法 | 优势 $A_i$ |
| --- | --- |
| GRPO | $(r_i - \bar r_g)\,/\,(\operatorname{std}(r_g) + \varepsilon)$ |
| Dr. GRPO | $r_i - \bar r_g$ |
| RLOO | $r_i - \frac{1}{G-1}\sum_{j \ne i} r_j$ |
| REINFORCE++-baseline | $x_i = r_i - \bar r_g$，再在整个 batch 上 $(x_i - \operatorname{mean}(x))\,/\,(\operatorname{std}(x) + \varepsilon)$ |

std 都用无偏估计（除以 $n-1$）。

**RLOO 与 Dr. GRPO 只差一个常数**：

$$
r_i - \bar r_g = \frac{G-1}{G}\Bigl(r_i - \frac{1}{G-1}\sum_{j \ne i} r_j\Bigr)
$$

所以两者方向完全相同，RLOO 等于 Dr. GRPO 乘 $\frac{G}{G-1}$。区别在理论上：组均值里含 $r_i$ 自己，严格说基线依赖了当前回答；留一法（leave-one-out）的基线不含它，梯度估计是无偏的。

## 形状

- `rewards` 是 `(B,)`，$B = P \cdot G$，同一提示的回答在连续位置上，`rewards.view(P, G)` 后每行是一组。
- 组内统计用 `dim=1`，REINFORCE++ 的白化在拉平后的 `(B,)` 上做。
- 输出 `(B,)`，用的时候再 `view(-1, 1)` 广播到 `(B, T)` 的 token 上。

## 用到的 PyTorch 操作

- `t.view(-1, G)`、`t.mean(dim=1, keepdim=True)`、`t.std(dim=1, keepdim=True)`：按组统计，`std` 默认无偏。
- `t.sum(dim=1, keepdim=True)`：按组求和，`keepdim` 让结果能广播回组内每个位置。留一法的基线是组内总和减去自己的奖励，再除以 $G-1$。
- `t.detach()`：优势是常数，不应该把梯度传回奖励。

## 常见坑

- **RLOO 的基线把自己也算进去**：那就成了 Dr. GRPO 的 $r_i - \bar r_g$，数值小了 $\frac{G-1}{G}$ 倍。
- **GRPO 用了总体标准差**：`unbiased=False` 会让结果和框架不一致，$G$ 越小差得越多。
- **REINFORCE++ 按组除标准差**：那就又回到 GRPO 了。这个方法的要点是只用组均值去中心化，缩放用整个 batch 共同的一个标准差。
- **REINFORCE++ 不减组均值，直接对原始奖励做全局白化**：这是不带 baseline 的 REINFORCE++，不同难度的题目没有各自的基准，简单题的回答会整体得到正优势。
- **eps 写死**：各框架的 eps 不一样（TRL 是 $10^{-4}$），要用传入的参数。

## 延伸

- **除以标准差带来的难度偏差**：0/1 奖励下，一组里答对比例为 $p$ 时，标准差约为 $\sqrt{p(1-p)}$。题目太难（$p$ 很小）或太简单（$p$ 很大）时标准差小，除完之后优势被放大。例如 $G=8$ 只有 1 个答对，它的优势约 $2.47$；4 个答对时每个答对的回答优势约 $0.94$。Dr. GRPO（Liu 等，2025）认为这相当于给题目隐式加了权重，主张去掉标准差。
- **部分得分的奖励更明显**：奖励是连续值时，一组回答的分数可能只差一点点，GRPO 会把这点微小差别放大成和别的组一样大的优势，噪声也被放大。全局白化（REINFORCE++）保留了不同组之间的相对大小。
- **为什么全局白化更稳**：整个 batch 只用一个缩放系数，优势的总体尺度固定，不会因为某组方差特别小而出现极端值，学习率也更好调。
- **RLOO 的来历**：留一法基线来自 Kool 等人 2019 年的 *Buy 4 REINFORCE Samples, Get a Baseline for Free!*；Ahmadian 等人 2024 年的 *Back to Basics* 把它用到 RLHF，指出 PPO 的很多部件在大模型上不必要。
- **其他基线**：ReMax 用贪心解码那个回答的奖励当基线；PPO 用 critic。本质都是找一个和当前回答无关、又尽量接近期望奖励的数。
- **全组同分**：全对或全错的组，四种方法的优势都是 $0$，这些采样白费了。DAPO 的动态采样会把这样的组过滤掉，见「DAPO 损失」。

## 面试追问

1. **为什么减去基线不改变梯度的期望？**
   $\mathbb{E}[b\, \nabla \log \pi(o)] = b\, \nabla \sum_o \pi(o) = b\, \nabla 1 = 0$，前提是 $b$ 与采样的 $o$ 无关。基线只影响方差。
2. **RLOO 和 GRPO 的区别？**
   RLOO 用组内其他回答的均值当基线，不除标准差，梯度无偏；GRPO 用含自己的组均值，再除以组内标准差。不看标准差这一步，RLOO 就是 Dr. GRPO 乘 $\frac{G}{G-1}$。
3. **GRPO 除以标准差有什么问题？**
   方差小的组（太难或太简单的题、奖励差别很小的组）优势被放大，相当于隐式地按题目难度加权，也会放大奖励噪声。Dr. GRPO 去掉了这一步，REINFORCE++ 改成全 batch 共用一个标准差。
4. **组大小 $G$ 怎么选？**
   $G$ 越大，基线估计越准、全组同分的概率越低，但每个提示的采样成本线性增加。常见取值 4 到 16；题目难度分布很偏时，$G$ 太小会让大量组全对或全错，没有梯度。
5. **这些方法还需要参考模型吗？**
   优势的计算不需要。参考模型只在要加 KL 约束时才用到，KL 可以写进损失（GRPO）或写进奖励（REINFORCE++ 原版的做法）。
