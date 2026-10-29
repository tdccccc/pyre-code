## 背景

ViT 的编码器块和 GPT-2 块结构几乎一样，都是 pre-norm 的「注意力 + MLP」，各带一个残差连接。区别在于：

- **没有因果掩码**：图像块之间没有先后顺序，每个块都能看到所有块（双向注意力）
- **CLS token**：序列开头加一个可学习的特殊 token，最后用它的输出做分类。所以序列长度是块数 $+ 1$

## 核心公式

**块结构**：

$$
x \leftarrow x + \operatorname{MHA}\bigl(\operatorname{LN}_1(x)\bigr),
\qquad
x \leftarrow x + \operatorname{MLP}\bigl(\operatorname{LN}_2(x)\bigr)
$$

**多头自注意力**（$h$ 个头，每头 $d_h = d / h$）：

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_h}}\right) V_i,
\qquad
\operatorname{MHA}(x) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

**MLP**：

$$
\operatorname{MLP}(x) = W_2\, \operatorname{GELU}(W_1 x + b_1) + b_2,
\qquad
\operatorname{GELU}(x) = \frac{x}{2}\left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

**融合的 QKV 投影**：用一个 `Linear(d, 3d)` 一次算出 Q、K、V，比三个独立的线性层更高效。输出的最后一维按「Q 的 $d$ 维、K 的 $d$ 维、V 的 $d$ 维」排列，每一段内再按头排列。

## 形状

| 步骤 | 形状 |
| --- | --- |
| 输入 $x$ | `(B, N, d)` |
| QKV 投影 | `(B, N, 3d)` |
| `reshape` | `(B, N, 3, h, d_h)` |
| `permute(2, 0, 3, 1, 4)` | `(3, B, h, N, d_h)`，取 `[0]`、`[1]`、`[2]` 得到 Q、K、V |
| 注意力分数 | `(B, h, N, N)` |
| 合头后 | `(B, N, d)` |
| MLP 隐藏层 | `(B, N, 4d)` |

## 用到的 PyTorch 操作

- `nn.LayerNorm(d)`、`nn.Linear`（允许使用）。
- `torch.erf`：手写 GELU。
- `reshape` / `permute` / `transpose`、`torch.softmax`。

## 常见坑

- **加了因果掩码**：ViT 是双向注意力。
- **调用了 `F.*`**：包括 `F.gelu`、`F.softmax` 等都不允许；`torch.softmax` 可以用。
- **QKV 拆分顺序与投影布局不一致**：`reshape(B, N, 3, h, d_h)` 中 `3` 必须在 `h` 之前，和「先 Q 再 K 再 V」的布局对应。
- **属性名**：测试会检查 `norm1`、`norm2` 是 `nn.LayerNorm`。
- **post-norm**：LayerNorm 应在子层之前。
