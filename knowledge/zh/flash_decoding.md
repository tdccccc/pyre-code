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

## 延伸

- **预填充与解码的差别**：

| | 预填充 | 解码 |
| --- | --- | --- |
| 每个头的查询数 | $S$ 个 | $1$ 个 |
| 可并行的维度 | batch、头、查询块 | batch、头 |
| 瓶颈 | 偏计算 | 访存 |

- **为什么解码是访存受限**：每个 key 和 value 各读一次，参与的运算只有一次点积和一次加权累加。以 bf16 存储，每读 $1$ 字节只有约 $1$ 次浮点运算，远低于 GPU 需要的几十到上百（roofline 的做法见「性能估算」专题）。所以解码的速度主要取决于从 HBM 读 KV 缓存的快慢，要让所有 SM 的读取带宽都用起来，就需要足够多的并行线程块。
- **两个阶段**：第一阶段每段一个线程块，各自算出 $o_c$ 和 $\operatorname{lse}_c$，写回 HBM；第二阶段一个小 kernel 读回所有分段结果，按 $e^{\operatorname{lse}_c - \operatorname{lse}}$ 加权求和。写回的数据量只有 $B \times H \times n \times (D + 1)$，相比读 KV 可以忽略。
- **分段数 $n$ 怎么定**：目标是让 $B \times H \times n$ 超过 SM 的数量，让所有 SM 都有活干。$n$ 太大则每段太短，第二阶段的归约和线程块的启动开销变得显著。$B \times H$ 本来就够大时，不需要切。FlashAttention 库在 `flash_attn_with_kvcache` 里有 `num_splits` 参数，默认由库按情况自动选。
- **lse 是有用的副产物**：它不只是为了合并。任何「把一份 softmax 拆成多份分别算再合并」的场景都靠它，比如 Ring Attention 里的跨设备合并（见「分布式训练」专题），或把共享前缀的注意力与各请求自己的后缀部分分开计算。
- **边界情况**：若某段全被掩码屏蔽，$\operatorname{lse}_c = -\infty$，合并时它的权重是 $0$，应当安全地被忽略；但段内 softmax 会变成 $0/0$，实现时要小心不让 NaN 混进结果。本题保证 $1 \le n \le S$，不会出现空段。
- **与 GQA 的配合**：GQA 下同一组的多个查询头共用一份 K、V，实现上可以让它们一起读 KV，只读一遍（见「注意力」专题的 GQA 一题）。GQA 本身也直接缩小了解码要读的 KV 总量。
- **本题的限制**：题目禁止构造 `(B, H, S)` 的完整分数，因为真实的 kernel 里每段的分数只存在于片上，从不整体出现。

## 面试追问

1. **解码时为什么 FlashAttention 的并行方式不够用？**
   它在 batch、头、查询块上并行。解码每个头只有一个查询，并行度只剩 $B \times H$，$B$ 小而 KV 很长时只有少数 SM 在工作。Flash-Decoding 增加了沿 KV 长度切分这一维。
2. **为什么各段合并的结果是精确的？**
   每段输出 $o_c$ 是段内 softmax 的结果，乘上 $e^{\operatorname{lse}_c - \operatorname{lse}}$ 后恰好变成这一段对全局 softmax 的贡献，$\sum_{j \in \mathcal{S}_c} e^{s_j - \operatorname{lse}} v_j$，各段相加就是完整的 softmax 加权和。
3. **Flash-Decoding 和 FlashAttention 的在线 softmax 是什么关系？**
   用的是同一个合并恒等式。FlashAttention 是串行地逐块合并，Flash-Decoding 是先并行算各段，再做一次归约，因为合并满足结合律，两种顺序结果一致。
4. **`num_splits` 越大越好吗？**
   不是。分段多可以让更多 SM 参与，但每段太短时，第二阶段的归约、中间结果的读写、线程块的启动开销会抵消收益。目标是刚好用满 GPU，$B \times H$ 已经够大时不需要切。
5. **为什么说解码是访存受限的？**
   每个 key、value 只读一次、只参与很少的运算，每字节的运算量远低于 GPU 的算力与带宽之比。所以缩小 KV 缓存（GQA、MLA、量化）能直接加快解码。
