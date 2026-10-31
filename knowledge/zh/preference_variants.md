## 背景

DPO（见「DPO 损失」一题）之后出现了一大批变体。它们用的数据和 DPO 一样（每个提示一对「选中 / 拒绝」回答），改的是三件事里的一件或几件：

- **损失的形状**：DPO 用 $-\log\sigma$，偏好越确定越想把间隔推到无穷大。IPO 换成平方损失，cDPO 加上标签平滑，让最优间隔变成有限值。
- **要不要参考模型**：SimPO、ORPO 去掉 $\pi_{\text{ref}}$，训练时少一次前向，也不用预先算参考对数概率。
- **对数概率要不要按长度归一化**：序列对数概率是 token 对数概率之和，回答越长越小。SimPO、ORPO 改用每个 token 的平均对数概率。

本题把 IPO、cDPO、SimPO、ORPO 四种放在一个函数里，正好覆盖这三个方向。

## 核心公式

记策略的序列对数概率为 $\pi_c$、$\pi_r$（选中、拒绝），参考模型的为 $\mathrm{ref}_c$、$\mathrm{ref}_r$，回答长度为 $|y_c|$、$|y_r|$。

**DPO 里的隐式奖励间隔**（IPO、cDPO 共用）：

$$
h = (\pi_c - \mathrm{ref}_c) - (\pi_r - \mathrm{ref}_r)
$$

**IPO**：让 $h$ 回归到一个有限的目标值，而不是越大越好：

$$
L_{\text{IPO}} = \operatorname{mean}\Bigl[\bigl(h - \tfrac{1}{2\beta}\bigr)^2\Bigr]
$$

$\beta$ 越大，目标间隔 $\frac{1}{2\beta}$ 越小，策略被拴得越紧，和 DPO 里 $\beta$ 的方向一致。

**cDPO**（conservative DPO）：假设每个偏好标签有 $\varepsilon$ 的概率被标反，对两种标签的 DPO 损失取期望，$z = \beta h$：

$$
L_{\text{cDPO}} = \operatorname{mean}\bigl[-(1-\varepsilon)\log\sigma(z) - \varepsilon\log\sigma(-z)\bigr]
$$

$\varepsilon = 0$ 就是 DPO。对 $z$ 求导令其为 $0$，得到最优点 $\sigma(z^*) = 1 - \varepsilon$，即 $z^* = \log\frac{1-\varepsilon}{\varepsilon}$，间隔不再趋于无穷。

**SimPO**：不要参考模型，用平均对数概率 $\bar a = \pi / |y|$ 当隐式奖励，并要求选中回答至少高出一个间隔 $\gamma$：

$$
L_{\text{SimPO}} = \operatorname{mean}\Bigl[-\log\sigma\bigl(\beta(\bar a_c - \bar a_r) - \gamma\bigr)\Bigr]
$$

**ORPO**：也不要参考模型，把 SFT 和偏好合成一个损失。令 $p = e^{\bar a}$ 为「平均每个 token 的概率」，它的对数赔率是

$$
\operatorname{lo}(\bar a) = \log\frac{p}{1-p} = \bar a - \log\bigl(1 - e^{\bar a}\bigr)
$$

$$
L_{\text{ORPO}} = \underbrace{\operatorname{mean}[-\bar a_c]}_{\text{选中回答的 SFT 损失}} + \lambda\,\operatorname{mean}\Bigl[-\log\sigma\bigl(\operatorname{lo}(\bar a_c) - \operatorname{lo}(\bar a_r)\bigr)\Bigr]
$$

本题用 `beta` 充当 $\lambda$。

## 形状

所有输入都是 `(B,)`：序列对数概率、长度、参考对数概率。输出是标量。长度可能是整数张量，做除法前转成浮点。

## 用到的 PyTorch 操作

- `F.logsigmoid(z)`：$\log\sigma(z)$ 的稳定写法，$z = -600$ 时也不会得到 $-\infty$。
- `torch.expm1(a)`：计算 $e^a - 1$，$a$ 接近 $0$ 时精度远高于 `torch.exp(a) - 1`。$\log(1 - e^a)$ 写成 `torch.log(-torch.expm1(a))`。
- `t.detach()`：参考模型的对数概率是常数。
- `lens.to(logps.dtype)`：整数长度转浮点。

## 常见坑

- **SimPO 用了求和的对数概率**：长回答的和天然更小，模型会被推向短回答，这正是 SimPO 要去掉的长度偏差。
- **IPO 把 $\beta$ 乘在间隔上**：写成 $(\beta h - \frac12)^2$ 后最优点没变，但损失和梯度都差了 $\beta^2$ 倍。
- **cDPO 的两项权重写反**：$1-\varepsilon$ 跟着「标签正确」的 $\log\sigma(z)$，$\varepsilon$ 跟着「标签被标反」的 $\log\sigma(-z)$。
- **ORPO 漏了 SFT 项**：只剩赔率比项时，模型只学「相对谁好」，不学生成选中回答本身。
- **$\log(1 - e^{a})$ 的数值问题**：float32 下 $a = -10^{-8}$ 时 `torch.exp(a)` 舍入成 $1.0$，`log1p(-exp(a))` 和 `log(1 - exp(a))` 都得到 $-\infty$，对数赔率变成 $+\infty$。这个值出现在拒绝回答一侧时损失直接变成 `inf`。
- `log(sigmoid(z))` 在 $z$ 很负时下溢为 $-\infty$，要用 `F.logsigmoid`。

## 延伸

- **为什么 DPO 会过拟合**：IPO 论文指出，偏好越接近确定（某一对总是 A 比 B 好），DPO 的最优解越要求 $h \to \infty$，KL 正则拦不住，策略会把拒绝回答的概率压到 $0$。IPO 用平方损失给间隔一个有限目标；cDPO 用标签平滑达到类似效果。
- **在 TRL 里的位置**（以较新的版本为准，和本题的差别也列在这里）：
  - cDPO 是 Mitchell 的 cDPO 笔记里给出的带标签平滑的 sigmoid 损失。TRL 现在的 `sigmoid` 损失不再做平滑，`DPOConfig.label_smoothing` 只被 `loss_type="robust"` 和 `"exo_pair"` 使用。`robust`（rDPO）同样针对噪声标签，但它构造的是无偏估计，公式和 cDPO 不同，别混为一谈。
  - IPO 是 `DPOTrainer` 的 `loss_type="ipo"`。TRL 会先把对数概率按回答长度平均再代入；本题按论文用求和的对数概率。
  - SimPO 在 `CPOTrainer` 里，`loss_type="simpo"`，间隔参数叫 `simpo_gamma`。`DPOTrainer` 里最接近 SimPO 长度归一化的是 `loss_type="sigmoid_norm"`。
  - ORPO 有单独的 `ORPOTrainer`，其中 `beta` 就是 $\lambda$。它的 SFT 项是整个 batch 上按 token 平均的交叉熵；本题先在每条回答内平均，再对 batch 平均，回答长度不同时两者不相等。
  - `CPOTrainer`、`ORPOTrainer`、`KTOTrainer` 在较新的 TRL 里已移到 `trl.experimental` 下。
- **SimPO 的超参**：$\beta$ 比 DPO 大得多（论文里多在 $2$ 到 $10$ 之间），因为平均对数概率的差值本身就小；$\gamma$ 通常按 $\gamma / \beta$ 的比例来调。
- **ORPO 为什么能跳过 SFT**：SFT 项让模型学会生成选中回答，赔率比项同时把拒绝回答压下去，一个阶段完成两件事。代价是没有参考模型拴着，偏好项的强度全靠 $\lambda$ 控制。
- **去掉参考模型的代价**：省显存、省一次前向，但失去了「离 SFT 模型多远」这个锚点，更依赖超参和数据质量。

## 面试追问

1. **IPO 为什么能缓解 DPO 的过拟合？**
   DPO 的 $-\log\sigma(\beta h)$ 只在 $h \to \infty$ 时趋于 $0$，偏好确定时会一直把间隔往大推；IPO 的平方损失在 $h = \frac{1}{2\beta}$ 处取到最小值 $0$，越过之后反而受罚，间隔被拴在有限值上。
2. **cDPO 的最优间隔是多少？**
   对 $z$ 求导得 $\sigma(z^*) = 1 - \varepsilon$，即 $z^* = \log\frac{1-\varepsilon}{\varepsilon}$。比如 $\varepsilon = 0.1$ 时 $z^* \approx 2.2$，不再趋于无穷。
3. **SimPO 为什么要按长度平均？**
   DPO 的隐式奖励用序列对数概率之和，而生成时模型按每个 token 的平均似然来「选」回答，两者不一致；和还会让长度主导奖励。按长度平均后，训练目标与生成时的度量一致，也减轻了长度偏差。
4. **SimPO 和 ORPO 都不需要参考模型，区别在哪？**
   SimPO 只有偏好项，通常仍从 SFT 模型开始训练；ORPO 把 SFT 的负对数似然和赔率比项加在一起，可以直接从基础模型一步训完。
5. **这些变体里哪个能直接复用 DPO 的数据管线？**
   全部可以，它们都吃同样的成对数据。差别只在损失和是否要预先算参考对数概率：IPO、cDPO 要，SimPO、ORPO 不要，但需要每个回答的长度。
