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

## 延伸

- **梯度在做什么**：令 $\hat r(y) = \beta \log(\pi_\theta(y) / \pi_{\text{ref}}(y))$ 为隐式奖励，则

$$
\nabla_\theta L = -\beta\, \mathbb{E}\Bigl[\sigma\bigl(\hat r_l - \hat r_w\bigr)\bigl(\nabla_\theta \log \pi_\theta(y_w) - \nabla_\theta \log \pi_\theta(y_l)\bigr)\Bigr]
$$

  前一项是权重：当隐式奖励把两者排反了（$\hat r_l > \hat r_w$）时权重大，已经排对的样本权重小。后一项是在提高选中回答的概率、降低拒绝回答的概率。
- **显存与吞吐**：只需要策略和参考模型，没有在线采样。参考模型的对数概率不变，可以在训练前预先算好存下来，训练时只加载策略。
- **序列对数概率**：`*_logps` 是回答里每个 token 的对数概率之和（prompt 部分不计入）。回答越长，和越小，所以 DPO 容易对长度敏感。
- **选中回答的概率也会下降**：损失只关心两者之差，把拒绝回答的概率压下去比抬高选中回答更容易，两者的概率可能一起下降。这是 DPO 的已知现象，也是一些变体要改进的地方。
- **离线的局限**：偏好数据来自别的模型，策略一旦偏离，数据就不再是它自己的分布。一个缓解办法是迭代做：用当前策略重新采样、标注，再训练一轮。
- **变体**（只记要点）：
  - IPO：把对数似然损失换成平方损失，抑制偏好确定时的过拟合；
  - KTO：不需要成对数据，只需要每个回答「好 / 坏」的标签；
  - SimPO：去掉参考模型，用长度归一化的平均对数概率当隐式奖励；
  - ORPO：把偏好项直接加进 SFT 损失，也不需要参考模型。

## 面试追问

1. **DPO 的推导里，$Z(x)$ 为什么能消掉？**
   $Z(x)$ 只依赖 prompt $x$，不依赖具体回答。Bradley-Terry 只看同一个 $x$ 下两个回答的奖励之差，相减时 $\beta \log Z(x)$ 刚好抵消，剩下策略与参考模型的对数概率比。
2. **$\beta$ 怎么理解？取大取小各有什么影响？**
   $\beta$ 是 KL 约束的强度。$\beta$ 大，策略被拴得更紧，更贴近参考模型；$\beta$ 小，策略可以偏离得更远，也更容易过拟合偏好数据。$0.1$ 是常用的默认值。
3. **DPO 相比 RLHF（PPO）的优缺点？**
   优点：不需要单独训练奖励模型、不需要在线采样和 critic，只要两个模型，稳定、简单、省显存。缺点：离线训练，受偏好数据分布限制；没有显式奖励模型，难以复用或检查；对长度、数据噪声比较敏感。
4. **DPO 里还有「奖励模型」吗？**
   没有单独的奖励模型，但策略本身隐含一个奖励 $\hat r(y) = \beta \log(\pi_\theta / \pi_{\text{ref}})$。训练时常用它监控「选中回答的隐式奖励是否高于拒绝回答」这类准确率。
5. **为什么训练时选中回答的对数概率也可能下降？**
   损失只约束差值：降低拒绝回答的概率比抬高选中回答更容易，参数共享时两者会一起变化，只要差值变大损失就下降，选中回答的绝对概率不一定上升。
