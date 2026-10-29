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
