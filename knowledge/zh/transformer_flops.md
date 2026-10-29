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
