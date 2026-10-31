## 背景

RMSNorm 是 LayerNorm 的简化版，LLaMA、Mistral 等现代大模型都用它。研究发现 LayerNorm 起作用的主要是「缩放到统一尺度」，而「减去均值」贡献不大。RMSNorm 去掉了减均值和偏置 $\beta$，计算更少，效果相当。

## 核心公式

对最后一维的向量 $x = (x_1, \dots, x_D)$，先算**均方根**（Root Mean Square）：

$$
\operatorname{RMS}(x) = \sqrt{\frac{1}{D}\sum_{i=1}^{D} x_i^2 + \epsilon}
$$

再逐元素缩放：

$$
y_i = \frac{x_i}{\operatorname{RMS}(x)} \cdot w_i
$$

$w$ 是长度为 $D$ 的可学习参数。和 LayerNorm 对照：

$$
\text{LayerNorm: } \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma + \beta
\qquad
\text{RMSNorm: } \frac{x}{\sqrt{\overline{x^2} + \epsilon}}\, w
$$

当 $\mu = 0$ 时，$\sigma^2 = \overline{x^2}$，两者（除了 $\beta$）完全一样。

## 形状

`x` 为 `(..., D)`，均方值沿最后一维计算得到 `(..., 1)`（保留维度），广播回 `(..., D)`；`weight` 为 `(D,)`。

## 用到的 PyTorch 操作

- `x.pow(2)` 或 `x * x`：逐元素平方。
- `.mean(dim=-1, keepdim=True)`。
- `torch.sqrt`，或 `torch.rsqrt` 直接得到 $1/\sqrt{\cdot}$ 再相乘。

## 常见坑

- **多减了均值**：那就变回 LayerNorm 了。
- **$\epsilon$ 放错位置**：本题的 $\epsilon$ 在根号里面，加在均方值上。
- **`keepdim` 漏掉**导致广播错位。
- **对 `x` 求了绝对值的均值**：RMS 是先平方、再平均、再开方。

## 延伸

- **省了什么**：和 LayerNorm 相比，少了一次求均值、少了一组参数 $\beta$。单个算子看省得不多，但它在每层出现两次，模型越大、序列越长，累计的节省越可观。RMSNorm 论文报告，在不同模型上整体运行时间减少约 $7\%$–$64\%$。
- **为什么不减均值也行**：LayerNorm 有两种不变性：对输入整体平移不变（减均值带来的），以及对输入整体缩放不变（除以标准差带来的）。实验表明起作用的主要是后者。RMSNorm 保留了缩放不变性：对任意 $\alpha > 0$（忽略 $\epsilon$），

$$
\operatorname{RMSNorm}(\alpha x) = \operatorname{RMSNorm}(x)
$$

- **数值精度**：半精度下 $x^2$ 很容易溢出，fp16 最大只能表示约 $65504$，$x = 300$ 时 $x^2$ 就超出了。LLaMA 等模型的实现都先把输入转成 fp32 算 RMS，归一化之后再转回原来的精度。
- **实现上的变体**：Gemma 用 $(1 + w)$ 作为缩放系数，$w$ 初始化为 $0$，这样初始时就是纯归一化。读别人的代码时注意区分。
- **QK-Norm**：Qwen3、Gemma 3 等模型在注意力里对每个头的 $q$、$k$ 再各做一次 RMSNorm，防止点积分数过大导致 softmax 饱和、训练不稳定。

## 面试追问

1. **RMSNorm 和 LayerNorm 有什么区别？什么情况下两者等价？**
   RMSNorm 不减均值、没有 $\beta$，只除以均方根。当输入均值为 $0$ 且 $\beta = 0$ 时，两者完全一样。
2. **RMSNorm 有平移不变性吗？**
   没有。给输入整体加一个常数，均方根会变，输出也会变。它只有缩放不变性。
3. **为什么现代大模型都用 RMSNorm？**
   效果和 LayerNorm 相当，但计算更少、参数更少；对访存受限的算子来说，少一次归约也意味着更好融合进其他 kernel。
4. **用 fp16 实现 RMSNorm 要注意什么？**
   平方求和可能溢出，应先转 fp32 计算均方根，最后再转回原精度。
5. **输入是 `(B, S, D)`，RMSNorm 有多少参数？**
   $D$ 个，只有缩放向量 $w$。
