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
- **属性名**：测试会检查 `norm1`、`norm2` 是 `nn.LayerNorm`，数值测试还会直接取 `qkv`、`proj`（注意力输出投影）、`fc1`、`fc2`（MLP 的两层）这几个属性。
- **post-norm**：LayerNorm 应在子层之前。

## 延伸

- **和 GPT-2 块的关系**：结构相同，只差一个因果掩码。GPT-2 Block 一题用掩码做自回归生成，ViT 不加掩码，是双向的编码器。两者都是 Pre-LN，堆叠若干块之后还要再接一个 LayerNorm，再接输出头（见「归一化」专题）。
- **参数量**：不计偏置和 LayerNorm，一个块约 $12d^2$ 个参数：QKV 投影 $3d^2$，输出投影 $d^2$，MLP 的两层各 $4d^2$。MLP 占了三分之二。ViT-B（$d=768$，$12$ 层）的编码器约 $12\times12\times768^2 \approx 8.5\times10^7$，对得上它常被引用的 $86\text{M}$ 量级。
- **计算量**：每个块的乘加量为

$$
\text{MACs} \approx 12\, N d^2 + 2\, N^2 d
$$

  第一项来自各个线性层，第二项来自 $QK^{\top}$ 和 $AV$。两项相等时 $N = 6d$。$224\times224$、$P=16$ 时 $N=197$，远小于 $6\times768$，线性层是大头，一个块约 $1.45\times10^9$ 次乘加；序列变长之后（高分辨率、小块），平方项才会成为瓶颈，那时要用 FlashAttention（见「IO 感知 kernel」专题）或窗口注意力（Swin）。
- **注意力的显存**：分数矩阵是 `(B, h, N, N)`，本题的朴素写法会把它整个存下来；这也是长序列时需要融合 kernel 的原因。
- **融合的 QKV**：一个 `Linear(d, 3d)` 比三个独立的线性层少两次 kernel 启动，输入只读一次，矩阵更大、GPU 利用率更高。
- **MLP 宽度**：隐层取 $4d$ 是沿用 Transformer 的惯例，不是推出来的。GPT-2 和原始 ViT 都用 $4d$。注意力头数需整除 $d$，ViT-B 是 $12$ 头、每头 $64$ 维。
- **CLS 与池化**：序列里多出的 CLS token 在块内和普通 token 一视同仁，区别只在最后是否取它做分类。两种取全局表示的做法见「卷积与视觉」专题。

## 面试追问

1. **ViT 块和 GPT-2 块有什么区别？**
   结构几乎一样（Pre-LN、注意力 + MLP、残差）。区别在于 ViT 没有因果掩码，是双向注意力；输入是 patch 嵌入加位置嵌入，而不是词嵌入；分类时读 CLS token 或平均池化，而不是预测下一个 token。
2. **一个 ViT 块有多少参数？计算量由什么主导？**
   约 $12d^2$ 个参数，其中注意力占 $4d^2$，MLP 占 $8d^2$。计算量约 $12Nd^2 + 2N^2d$，$N$ 小于 $6d$ 时线性层主导，$N$ 很大时注意力的平方项主导。
3. **为什么 ViT 的注意力在高分辨率下会成为瓶颈？有什么办法？**
   token 数 $N$ 随图像面积增长，注意力的计算和显存是 $N^2$。办法有：增大块大小减少 token，用窗口注意力限制范围，或用 FlashAttention 这类融合 kernel 省显存。
4. **为什么用融合的 QKV 投影？**
   数学上与三个独立线性层等价，但是一次大矩阵乘，kernel 启动更少、输入只读一次，更利于 GPU 吞吐。
5. **ViT 为什么在小数据集上从头训练不如 CNN？**
   它没有局部性和平移等变的先验，要从数据里学出来，所以更依赖大规模预训练或强数据增强（DeiT 就是靠这些只用 ImageNet 训练）。详见「卷积与视觉」专题的归纳偏置一节。
