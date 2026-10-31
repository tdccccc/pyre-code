## 背景

训练一个模型要多少卡、多少天？在动手之前，AI Infra 工程师会先估算两个数：**参数量**（决定显存）和 **FLOPs**（浮点运算次数，决定算力和时间）。估算结果再和硬件的实际吞吐对比，得到 **MFU**（Model FLOPs Utilization，模型算力利用率），用来衡量训练系统的效率。

## 核心公式

**矩阵乘的 FLOPs**：$A \in \mathbb{R}^{m \times k}$ 乘 $B \in \mathbb{R}^{k \times n}$，输出有 $mn$ 个元素，每个是 $k$ 次乘加：

$$
\text{FLOPs}(AB) = 2\, m\, k\, n
$$

一次乘加计为 $2$ FLOPs（一次乘法 + 一次加法）。

**一个线性层**：$t$ 个 token 通过一个 $d_{\text{in}} \times d_{\text{out}}$ 的权重，FLOPs 为 $2 \cdot t \cdot d_{\text{in}} d_{\text{out}} = 2 \cdot t \cdot (\text{该层参数量})$。所以权重矩阵乘部分的 FLOPs ≈ $2 \times \text{参数量} \times \text{token 数}$。

**每层参数**（$d$ = `d_model`）：

| 部分 | 参数量 |
| --- | --- |
| $W_Q, W_K, W_V, W_O$ | $4d^2$ |
| MLP 上投影 + 下投影 | $2\, d\, d_{ff}$ |
| 两个 LayerNorm（各有 $\gamma, \beta$） | $4d$ |

**注意力本身的矩阵乘**不含参数，单独计算。每个序列：

$$
\underbrace{Q K^{\top}}_{(s \times d)(d \times s)} :\ 2 s^2 d,
\qquad
\underbrace{A V}_{(s \times s)(s \times d)} :\ 2 s^2 d
$$

合计每层 $4\, b\, s^2 d$。这一项随序列长度**平方**增长，长上下文时不可忽略。

**反向是前向的 2 倍**：每个线性层 $Y = XW$ 反向时要算两个同样大小的矩阵乘：

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^{\top},
\qquad
\frac{\partial L}{\partial W} = X^{\top} \frac{\partial L}{\partial Y}
$$

所以训练 ≈ 前向 + 2 × 前向 = 3 × 前向，于是得到经验法则：

$$
\text{训练 FLOPs} \approx 6 \times N_{\text{params}} \times N_{\text{tokens}}
$$

**MFU**：

$$
\text{MFU} = \frac{\text{模型实际需要的 FLOPs} / \text{实际耗时}}{\text{硬件峰值 FLOP/s}}
$$

## 常见坑

- **共享权重时漏算 LM head 的计算**：共享只省参数，LM head 的矩阵乘每个 token 都要做一次。
- **把 embedding 查表算进 FLOPs**：查表不是矩阵乘，不产生 FLOPs；但它的参数要算进参数量。
- **LayerNorm 参数**：每层两个，每个 $2d$；最后还有一个最终 LayerNorm。
- **返回浮点数**：题目要求 `int`，全程用整数运算，不要出现 `/`。
- 本题不考虑因果掩码减半，也没有偏置。

## 延伸

**6ND 的偏差来自哪里**。题目的精确公式和 $6ND$ 差两处：

- 注意力的矩阵乘不含参数，不在 $N$ 里。MLP 隐藏维取 $4d$ 时，每层每个 token 的权重矩阵乘是 $2 \times 12 d^2 = 24 d^2$ FLOPs，注意力的 $QK^\top$ 与 $AV$ 是 $4 s d$，二者之比为：

$$
\frac{4sd}{24d^2} = \frac{s}{6d}
$$

序列长度接近或超过 $6d$ 时，注意力项就不能忽略。GPT-2 small（$d=768$，$s=1024$）的注意力项占前向总 FLOPs 约 $13\%$，$6ND$ 因此比精确值低约 $13\%$。

- 嵌入表的计数。权重共享时，同一个 $V \times d$ 矩阵既是 embedding 又是 LM head，$6N$ 把它算一次，正好等于 LM head 的矩阵乘；不共享时，输入 embedding 的 $Vd$ 个参数只是查表，却被 $6N$ 当作矩阵乘计入，会略微高估。小模型里词表占比很大（GPT-2 small 的嵌入表是总参数的约 $31\%$），大模型里这一项占比小得多。

**因果掩码**。本题按完整 $s \times s$ 计注意力。因果注意力只需要下三角，真正的 kernel 约只算一半。通常报告 MFU 时仍按完整 $s \times s$ 的「模型 FLOPs」计，这样不同实现之间可以比较。

**MFU 与 HFU**。MFU 的分子是模型本身需要的 FLOPs，不管实现怎么做；HFU（Hardware FLOPs Utilization）的分子是硬件实际执行的 FLOPs，会把激活重计算多做的前向也算进去。开全量重计算时，实际执行约是 $4\times$ 前向，即 $8ND$，HFU 比 MFU 高约三分之一，但训练并没有变快。

**用来估算时间**：

$$
T \approx \frac{6\, N D}{n_{\text{gpu}} \cdot P_{\text{峰值}} \cdot \text{MFU}}
$$

$n_{\text{gpu}}$ 是卡数，$P_{\text{峰值}}$ 是单卡峰值 FLOP/s。MFU 取一个经验值，大规模训练实测常在三四成上下，具体取决于模型、并行策略和通信开销。给定固定算力，Chinchilla 论文（2022）给出的经验是训练 token 数约为参数量的 $20$ 倍时最划算，此时 $D \approx 20N$，总 FLOPs 约 $120 N^2$。

**被忽略的项**。softmax、归一化、GELU、残差加法的 FLOPs 很少，但它们是逐元素操作，每个数据读写一遍只做很少的运算，本身受带宽限制，在墙钟时间里的占比可能比 FLOPs 占比大得多。这也是融合 kernel 能显著提速的原因。

## 面试追问

1. **训练 FLOPs 为什么约是 $6ND$？**
   每个参数对每个 token 前向参与一次乘加，记 2 FLOPs；反向对线性层 $Y = XW$ 要算两个同样大小的矩阵乘，分别求对 $X$ 和对 $W$ 的梯度，共 4 FLOPs。合计 $2 + 4 = 6$ 乘以参数量和 token 数。
2. **注意力项什么时候不可忽略？**
   每层每个 token 的注意力 FLOPs 约 $4sd$，权重矩阵乘约 $24d^2$（MLP 隐藏维为 $4d$ 时），比值是 $s/(6d)$。序列长度与 $6d$ 同量级甚至更长时，注意力项占比显著，长上下文训练必须计入。
3. **MFU 和 HFU 有什么区别？**
   MFU 只算模型必需的 FLOPs，HFU 还算上重计算等实现带来的额外 FLOPs。开了激活重计算后 HFU 变高，但墙钟时间没有缩短，所以对比不同系统时用 MFU 更公平。
4. **权重共享（tied embeddings）省了什么，没省什么？**
   省的是参数量和对应的显存、梯度、优化器状态，少了 $Vd$ 个参数。LM head 的矩阵乘每个 token 都要做，FLOPs 一点没少。
5. **MoE 模型怎么套 $6ND$？**
   $N$ 应取每个 token 实际激活的参数量（路由到的专家加共享部分），而不是总参数量。总参数量决定显存，激活参数量决定算力。
