## 背景

GPT-2 由若干个结构相同的 Transformer 块堆叠而成。每个块有两个子层：

1. **因果自注意力**：每个 token 汇总它自己和前面所有 token 的信息
2. **MLP**：对每个 token 单独做非线性变换

每个子层外面包着**残差连接**和 **LayerNorm**。这道题把前面学过的组件（LayerNorm、多头注意力、GELU、线性层）组装起来。

## 核心公式

**Pre-norm 残差结构**（GPT-2 把 LayerNorm 放在子层之前）：

$$
x \leftarrow x + \operatorname{Attn}\bigl(\operatorname{LN}_1(x)\bigr)
$$

$$
x \leftarrow x + \operatorname{MLP}\bigl(\operatorname{LN}_2(x)\bigr)
$$

残差让梯度可以沿 $x$ 直接流回浅层：$\frac{\partial}{\partial x}\bigl(x + f(x)\bigr) = I + \frac{\partial f}{\partial x}$，恒等项保证梯度不会消失。

**MLP**：

$$
\operatorname{MLP}(x) = W_2\, \operatorname{GELU}(W_1 x + b_1) + b_2,
\qquad W_1: d \to 4d,\ W_2: 4d \to d
$$

**因果掩码**：语言模型按顺序生成，位置 $i$ 只能看见位置 $j \le i$。在 softmax 之前把 $j > i$ 的分数设为 $-\infty$：

$$
S_{ij} \leftarrow
\begin{cases}
S_{ij}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

因为 $e^{-\infty} = 0$，这些位置的注意力权重恰好为 $0$，同时每行剩下的权重仍然和为 $1$。掩码是一个上三角矩阵（不含对角线），例如 $S = 4$：

$$
M = \begin{pmatrix}
0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0
\end{pmatrix}
\quad (1 \text{ 表示屏蔽})
$$

## 形状

输入输出都是 `(B, S, d_model)`。注意力内部的拆头与多头注意力一题相同；分数矩阵为 `(B, h, S, S)`，掩码 `(S, S)` 通过广播作用到所有 batch 和头。

## 用到的 PyTorch 操作

- `nn.LayerNorm(d_model)`、`nn.Linear`、`nn.Sequential`。
- `torch.triu(t, diagonal=1)`：取上三角，`diagonal=1` 表示不含主对角线。
- `scores.masked_fill(mask, float('-inf'))`：`mask` 为 `True` 的位置填入 $-\infty$。
- GELU：`F.gelu`、`nn.GELU`，或者用 `torch.erf` 自己写。

## 常见坑

- **掩码包含了对角线**：每个位置必须能看见自己，否则第一行全是 $-\infty$，softmax 得到 NaN。
- **写成 post-norm**：$\operatorname{LN}(x + \operatorname{Attn}(x))$ 是原始 Transformer 的做法，和本题要求不符。
- **残差加的是归一化后的值**：应是 $x + \operatorname{Attn}(\operatorname{LN}(x))$，而不是 $\operatorname{LN}(x) + \operatorname{Attn}(\operatorname{LN}(x))$。
- **两个子层共用一个 LayerNorm**：`ln1`、`ln2` 是两组独立参数。
- 测试会检查因果性：改动后面的 token，前面位置的输出不能变。
