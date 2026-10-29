## 背景

FlashAttention 在 batch、头、**查询块**三个方向上并行。但解码时每个头只有 $1$ 个查询，并行度只剩 $B \times H$。比如 $B = 1$、$H = 32$，只能启动 32 个线程块，而 A100 有 108 个 SM，大部分闲着。偏偏解码时 KV 缓存又很长，每个线程块要扫完整条序列。

**Flash-Decoding** 增加一个并行维度：把 KV 序列切成 $n$ 段，各段**同时**算出局部结果，最后再精确地合并。

## 核心公式

第 $c$ 段包含位置集合 $\mathcal{S}_c$，缩放分数 $s_j = q \cdot k_j / \sqrt{D}$。每段算两个量：

$$
\operatorname{lse}_c = \log \sum_{j \in \mathcal{S}_c} e^{s_j},
\qquad
o_c = \sum_{j \in \mathcal{S}_c} e^{s_j - \operatorname{lse}_c}\, v_j
$$

$o_c$ 是只在本段内做 softmax 得到的输出。

**合并**：

$$
\operatorname{lse} = \log \sum_{c} e^{\operatorname{lse}_c},
\qquad
o = \sum_{c} e^{\operatorname{lse}_c - \operatorname{lse}}\, o_c
$$

**为什么精确**：代入 $o_c$，

$$
e^{\operatorname{lse}_c - \operatorname{lse}}\, o_c = \sum_{j \in \mathcal{S}_c} e^{s_j - \operatorname{lse}}\, v_j
$$

对所有段求和就是 $\sum_{j} e^{s_j - \operatorname{lse}}\, v_j$，而 $e^{\operatorname{lse}} = \sum_j e^{s_j}$ 正是全局 softmax 的分母。权重 $e^{\operatorname{lse}_c - \operatorname{lse}}$ 就是第 $c$ 段在总 softmax 质量中所占的比例，各段比例之和为 $1$。

**和 FlashAttention 的关系**：两者用的是同一个恒等式。FlashAttention 的在线 softmax 是**串行**地逐块合并；Flash-Decoding 是先**并行**地算各块，再做一次归约。

**数值稳定**：$\operatorname{lse}$ 要用减最大值的形式计算：$\log \sum_j e^{s_j} = m + \log \sum_j e^{s_j - m}$。

## 形状

| 量 | 形状 |
| --- | --- |
| `q` | `(B, H, D)` |
| `K`、`V` | `(B, H, S, D)` |
| 第 $c$ 段的 $K_c$、$V_c$ | `(B, H, len_c, D)` |
| 第 $c$ 段分数 | `(B, H, len_c)` |
| $\operatorname{lse}_c$、$\operatorname{lse}$ | `(B, H)` |
| $o_c$、$o$ | `(B, H, D)` |

`torch.tensor_split` 切出的各段长度可以不同：前 $S \bmod n$ 段各多一个 token。

## 用到的 PyTorch 操作

- `torch.tensor_split(K, n, dim=2)`：切成 $n$ 段（允许不整除）。
- `torch.einsum('bhd,bhsd->bhs', q, Kc)`：每个头的查询与该段所有键做点积；`'bhs,bhsd->bhd'` 做加权求和。
- `torch.logsumexp(t, dim=...)`：稳定的 LSE，既用于段内，也用于合并。
- `torch.stack(list, dim=0)`。

## 常见坑

- **构造完整的 `(B, H, S)` 分数**：比如先把各段分数拼起来再统一 softmax。结果正确，但违背了本题的目的，测试会记录所有中间张量的形状。
- **合并时直接平均各段输出**：各段的 softmax 质量不同，权重必须是 $e^{\operatorname{lse}_c - \operatorname{lse}}$。
- **手写 $\log \sum e^{x}$ 不减最大值**：`q` 很大时溢出为 `inf`。
- **返回的 lse 只是某一段的**：要返回对全部 $S$ 个分数的 LSE。
- 缩放因子 $1/\sqrt{D}$ 要在 LSE 之前乘上。
