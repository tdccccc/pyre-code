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

## 延伸

- **参数量**：忽略偏置和 LayerNorm，注意力的 $W_Q, W_K, W_V, W_O$ 共 $4d^2$，MLP 的两个矩阵共 $2 \cdot d \cdot 4d = 8d^2$，一个块约 $12d^2$ 个参数，其中 MLP 占三分之二。以 GPT-2 small（12 层、$d = 768$、12 个头）为例，$12 \times 12 \times 768^2 \approx 85\text{M}$，再加上词嵌入（词表 $50257$）和位置嵌入，总共约 $124\text{M}$。
- **GPT-2 原版的几个细节**：位置信息用可学习的绝对位置嵌入，加在输入上，块内没有位置编码；GELU 用 tanh 近似；所有线性层带偏置；输出头与词嵌入共享权重；最后一个块之后还有一个 final LN。本题只实现块本身。
- **残差分支的初始化**：残差流上每层都会加一个分支的输出，层数多了方差会累积。GPT-2 论文写到，初始化时把残差分支输出层的权重按残差层数 $N$ 缩小为原来的 $1/\sqrt{N}$。初始化的一般原则见「基础层与激活」专题的权重初始化一题。
- **并行块**：GPT-J、PaLM 把两个子层改成并行，$x \leftarrow x + \operatorname{Attn}(\operatorname{LN}(x)) + \operatorname{MLP}(\operatorname{LN}(x))$，两个分支可以合并矩阵乘法，训练时略快。大多数模型仍用本题的串行写法。
- **推理时的复用**：因果掩码意味着旧 token 的 $K$、$V$ 不会因新 token 而改变，可以缓存起来，每步只算新 token，这就是 KV 缓存（见「推理服务」专题）。
- **和 LLaMA 式 block 的差别**：Pre-RMSNorm、RoPE、GQA、SwiGLU，逐项对比见「模型架构」专题。

## 面试追问

1. **一个 GPT-2 块有多少参数？注意力和 MLP 各占多少？**
   约 $12d^2$，注意力 $4d^2$、MLP $8d^2$，MLP 占三分之二。所以大模型的参数大头在 MLP，这也是 MoE 选择去替换 MLP 的原因。
2. **为什么 MLP 的隐层是 $4d$？**
   沿用了原始 Transformer 的 4 倍设定，是经验选择，不是推导出来的。LLaMA 用 SwiGLU 时有三个矩阵，隐层改成约 $\frac{8}{3}d$，让每层参数量和 $4d$ 的两矩阵 MLP 保持一致。
3. **残差连接起什么作用？**
   反向传播时 $\frac{\partial}{\partial x}(x + f(x)) = I + \frac{\partial f}{\partial x}$，恒等项让梯度能直达浅层，深层网络才训得动。另一种看法是把 $x$ 当作一条「残差流」，每个子层只往里读、往里加，不改写整条流。
4. **Pre-LN 和 Post-LN 的区别？为什么 GPT-2 用 Pre-LN？**
   Pre-LN 的残差通路上没有 LN，深层训练更稳；代价是需要在最后补一个 final LN。详见「归一化」专题。
5. **因果掩码为什么要用 $-\infty$，而不是直接把权重置 $0$？**
   在 softmax 之前屏蔽，剩下位置的权重才仍然和为 $1$；softmax 之后再置零会破坏归一化。每行至少要保留对角线，否则整行是 $-\infty$，softmax 会得到 NaN。
