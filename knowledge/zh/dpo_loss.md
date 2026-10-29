## 背景

RLHF 要先训练奖励模型，再用 PPO 做强化学习，流程长、超参多、训练不稳定。**DPO**（Direct Preference Optimization）发现：RLHF 目标的最优解可以写成闭式，于是能把奖励模型「代换」掉，直接用偏好数据通过一个分类式的损失训练语言模型。不需要奖励模型，也不需要采样和强化学习。

## 核心公式

**RLHF 的目标**：最大化奖励，同时不要偏离参考模型 $\pi_{\text{ref}}$（通常是 SFT 模型）太远：

$$
\max_{\pi}\ \mathbb{E}_{y \sim \pi}\bigl[r(x, y)\bigr] - \beta\, \operatorname{KL}\bigl(\pi(\cdot \mid x)\ \Vert\ \pi_{\text{ref}}(\cdot \mid x)\bigr)
$$

**它的最优解**：

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\left(\frac{r(x, y)}{\beta}\right)
$$

$Z(x)$ 是归一化常数。反过来，奖励可以用策略表示：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**代入 Bradley-Terry 偏好模型** $P(y_w \succ y_l) = \sigma(r_w - r_l)$：两个回答的 $\beta \log Z(x)$ 相同，相减后**抵消**，只剩下策略和参考模型的对数概率：

$$
L_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)\right]
$$

用对数概率写：

$$
L = -\operatorname{mean}\Bigl[\log \sigma\Bigl(\beta\bigl((\pi_c - \text{ref}_c) - (\pi_r - \text{ref}_r)\bigr)\Bigr)\Bigr]
$$

**隐式奖励**：$\hat{r}(y) = \beta \bigl(\log \pi_\theta(y) - \log \pi_{\text{ref}}(y)\bigr)$。DPO 让选中回答的隐式奖励高于拒绝回答。

**$\beta$ 的作用**：$\beta$ 越大，KL 约束越强，策略越贴近参考模型；在损失里它放大了对数概率差，使 sigmoid 更快饱和。

这里的「对数概率」是整个回答序列的对数概率，即回答中所有 token 对数概率之和。

## 形状

四个输入都是 `(B,)`，输出是标量。

## 用到的 PyTorch 操作

- 逐元素减法、乘以标量。
- `torch.sigmoid`、`torch.log`，或更稳定的 `F.logsigmoid`。
- `.mean()`。

## 常见坑

- **符号**：是「选中的比值 − 拒绝的比值」，$\beta$ 乘在整个差上。
- **漏了参考模型**：只用 $\pi_c - \pi_r$ 就失去了 KL 约束的含义。
- **$\log(\sigma(x))$ 在 $x$ 很负时得到 $-\infty$**：可以用 `F.logsigmoid` 避免。
- 参考模型的对数概率是常数，梯度只流向策略的对数概率。
