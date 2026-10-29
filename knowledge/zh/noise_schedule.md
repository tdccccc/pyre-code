## 背景

扩散模型的思路：先定义一个**加噪过程**，在 $T$ 步内逐渐往数据里加高斯噪声，直到变成纯噪声；再训练一个网络学习**反过来**一步步去噪。生成时从纯噪声出发，反复去噪就得到新样本。

**噪声调度**规定每一步加多少噪声。它决定了训练时各个噪声水平的分布，对生成质量影响很大。

## 核心公式

**单步加噪**：$\beta_t$ 是第 $t$ 步的噪声方差，$\alpha_t = 1 - \beta_t$：

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1 - \alpha_t}\, \epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0, I)
$$

**一步跳到任意时刻**：令累积乘积

$$
\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s)
$$

则

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon
$$

可以用归纳法验证方差：设 $x_{t-1}$ 的噪声部分方差为 $1 - \bar{\alpha}_{t-1}$，则 $x_t$ 的噪声方差为 $\alpha_t (1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t) = 1 - \bar{\alpha}_t$。训练时不需要一步步加噪，直接按这个公式采样任意 $t$。

$\bar{\alpha}_t$ 是**信号保留的比例**：接近 $1$ 时几乎是原图，接近 $0$ 时几乎是纯噪声。**信噪比** $\operatorname{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$。

**线性调度**：$\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$：

$$
\beta_t = \beta_{\text{start}} + (\beta_{\text{end}} - \beta_{\text{start}}) \frac{t}{T - 1}
$$

问题是 $\bar{\alpha}_t$ 下降太快，后面相当多的步都已接近纯噪声，训练资源被浪费。

**余弦调度**（Improved DDPM）：直接规定 $\bar{\alpha}_t$ 的形状，让它更平缓地下降：

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)},
\qquad
f(t) = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right),\quad s = 0.008
$$

除以 $f(0)$ 保证 $\bar{\alpha}_0 = 1$；小偏移 $s$ 防止开头的 $\beta$ 太小。

**sigmoid 调度**：$\beta_t$ 按 S 形曲线从 $\beta_{\text{start}}$ 过渡到 $\beta_{\text{end}}$，中间变化快、两端变化慢。

## 用到的 PyTorch 操作

- `torch.arange(T, dtype=torch.float32)`。
- `torch.cumprod(t, dim=0)`：累积乘积，$[a, b, c] \to [a,\ ab,\ abc]$。
- `torch.cos`、`math.pi`、`t.clamp(lo, hi)`。
- 把 $[\min, \max]$ 线性映射到 $[a, b]$：$a + (b - a)\dfrac{x - \min}{\max - \min}$。

## 常见坑

- **线性调度分母用了 $T$**：本题是 $t / (T - 1)$，保证最后一步恰好是 $\beta_{\text{end}}$。
- **用了 `cumsum`**：$\bar{\alpha}$ 是 $1 - \beta$ 的累积**乘积**。
- **余弦调度忘了除以 $f(0)$**，或者忘了截断到 $[0.0001, 0.9999]$。
- **sigmoid 调度没有缩放**：sigmoid 输出在 $(0, 1)$ 之间，要先线性映射到 $[\beta_{\text{start}}, \beta_{\text{end}}]$。
- 未知的 `schedule_type` 应报错。
