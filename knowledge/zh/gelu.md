## 背景

GELU（Gaussian Error Linear Unit）是 BERT、GPT 系列使用的激活函数。ReLU 在 $0$ 处是一个硬拐点：负数全部清零，正数原样通过。GELU 换成一个平滑的「软开关」：输入越大，越可能被保留。

## 核心公式

用标准正态分布的累积分布函数（CDF）$\Phi(x)$ 作为保留比例：

$$
\mathrm{GELU}(x) = x \cdot \Phi(x), \qquad \Phi(x) = P(Z \le x),\ Z \sim \mathcal{N}(0, 1)
$$

$\Phi(x)$ 从 $0$（$x \to -\infty$）平滑上升到 $1$（$x \to +\infty$），$\Phi(0) = \tfrac12$。

$\Phi$ 没有初等函数表达式，要借助**误差函数** $\operatorname{erf}$：

$$
\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^{z} e^{-t^2}\, \mathrm{d}t,
\qquad
\Phi(x) = \frac{1}{2}\left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

代入就得到题目要求的精确形式：

$$
\mathrm{GELU}(x) = \frac{x}{2}\left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

早期为了计算快，还有一个 $\tanh$ 近似版：

$$
\mathrm{GELU}(x) \approx \frac{x}{2}\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,\bigl(x + 0.044715\,x^3\bigr)\right)\right)
$$

## 用到的 PyTorch 操作

- `torch.erf(t)`：对张量逐元素计算 $\operatorname{erf}$，支持反向传播。
- `math.sqrt(2.0)`：$\sqrt{2}$ 只是一个 Python 浮点常数，用 `math` 计算即可。
- 加、乘、除都是逐元素运算，输出形状和输入相同。

## 常见坑

- **用 `math.erf` 处理张量**：`math.erf` 只接受单个 Python 数，传张量会报错，要用 `torch.erf`。
- **误用近似版**：题目要求和 `F.gelu`（默认精确版）的误差在 $10^{-4}$ 以内，$\tanh$ 近似在部分输入上可能超出。
- **忘了乘 $x$**：$\Phi(x)$ 只是保留比例，结果是 $x \cdot \Phi(x)$。
- 可以自检：$\mathrm{GELU}(0) = 0$；$x$ 很大时 $\mathrm{GELU}(x) \approx x$；$x$ 很小（很负）时趋近 $0$。

## 延伸

- **随机门控的期望**：以概率 $\Phi(x)$ 保留 $x$、否则置零。输入越大越可能被保留，输出的期望正好是

$$
\mathbb{E}\bigl[m \cdot x\bigr] = \Phi(x)\, x,
\qquad m \sim \operatorname{Bernoulli}\bigl(\Phi(x)\bigr)
$$

  所以 GELU 可以看作「随机门控」取期望的结果；ReLU 则是把门固定成确定的 $\mathbb{1}[x > 0]$。
- **导数**：令 $\varphi(x)$ 为标准正态的概率密度，则

$$
\mathrm{GELU}'(x) = \Phi(x) + x\, \varphi(x),
\qquad
\varphi(x) = \frac{1}{\sqrt{2\pi}}\, e^{-x^2/2}
$$

  $\mathrm{GELU}'(0) = \tfrac12$，处处连续可导，没有 ReLU 在 $0$ 处的拐点。
- **非单调**：负半轴有一小段负值，最低点在 $x \approx -0.75$、值约 $-0.17$；$x$ 更负时再回到 $0$。
- **两个常见近似**：
  - $\tanh$ 版（见上文），最大误差约 $5 \times 10^{-4}$，所以不满足本题 $10^{-4}$ 的容差。GPT-2 等早期实现用的就是这个近似，PyTorch 里写作 `F.gelu(x, approximate='tanh')`。
  - $x\,\sigma(1.702\,x)$：只用一个 sigmoid，更粗糙，最大误差在 $10^{-2}$ 量级。
- **精确版与近似版要对齐**：如果用别人训练好的权重，激活函数的版本要和训练时一致。把 tanh 近似换成精确版，单层误差很小，但整个模型的输出会有细小而可观测的偏差。
- **和 SiLU 的关系**：GELU 近似于 $x\,\sigma(1.702\,x)$，而 SiLU 是 $x\,\sigma(x)$，两者形状接近，只是曲线在 $0$ 附近的陡峭程度不同。
- **访存受限**：GELU 是逐元素运算，计算量和 FFN 的矩阵乘相比可以忽略，但每次都要读写整张大小为 `(B, S, d_ff)` 的中间张量，主要开销在访存。生产代码里通常把它和前后的算子融合，避免来回搬运。半精度下内部通常用更高精度计算。

## 面试追问

1. **GELU 和 ReLU 有什么区别？**
   ReLU 是硬门控，$x>0$ 时通过、否则清零，$0$ 处不可导；GELU 用 $\Phi(x)$ 做平滑门控，处处可导，负半轴有一小段非零输出，非单调。
2. **GELU 的直观含义是什么？**
   用输入自身的大小决定保留多少：$x\,\Phi(x)$ 是「以概率 $\Phi(x)$ 保留 $x$」的期望。输入越大越接近原样输出，越负越接近 $0$。
3. **GELU 和 Swish / SiLU 有什么关系？**
   GELU 近似为 $x\,\sigma(1.702\,x)$，SiLU 是 $x\,\sigma(x)$，形状很像。Swish 是带参数的版本 $x\,\sigma(\beta x)$，$\beta=1$ 时就是 SiLU。
4. **精确版和 tanh 近似版会差多少？什么时候要注意？**
   逐点最大误差约 $5\times10^{-4}$，数值上很小。但加载预训练权重时，激活函数要和训练时用的版本一致，否则会有细小但可观测的偏差。
5. **GELU 的导数是什么？在 $0$ 处是多少？**
   $\Phi(x) + x\,\varphi(x)$，$x=0$ 时为 $0.5$。
