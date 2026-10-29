## 背景

一个 batch 里的序列长短不一。传统做法是用 padding 补齐到最长的长度，例如长度 $[100, 2000, 300]$ 补齐后有 $3 \times 2000 = 6000$ 个位置，真实 token 只有 $2400$ 个，$60\%$ 的计算浪费在 padding 上。

**打包**（packing）：把所有序列首尾相接拼成一条，不做任何补齐，总长 $T = \sum_j \ell_j$。再用一个边界数组告诉注意力：哪些 token 属于同一个序列。FlashAttention 的 `flash_attn_varlen_func` 就是这个接口，训练时把短文档打包是标准做法。

## 核心公式

**累积边界** `cu_seqlens`（前缀和）：

$$
c_0 = 0,\qquad c_{j+1} = c_j + \ell_j,\qquad c_N = T
$$

第 $j$ 个序列占据打包后的位置 $[c_j,\ c_{j+1})$。

**每个 token 属于哪个序列**：token $t$ 的序列编号是满足 $c_j \le t$ 的最大 $j$：

$$
\operatorname{seg}(t) = \max\{\, j : c_j \le t \,\}
$$

**掩码**：token $i$ 只能看同一序列内的 token $j$（因果时还要 $j \le i$）：

$$
\text{允许}(i, j) = \bigl[\operatorname{seg}(i) = \operatorname{seg}(j)\bigr] \wedge \bigl[\,j \le i\,\bigr]
$$

长度 $[2, 3]$、`cu_seqlens = [0, 2, 5]` 的非因果掩码是**块对角**的（$\cdot$ 可见，$\times$ 屏蔽）：

$$
\begin{pmatrix}
\cdot & \cdot & \times & \times & \times \\
\cdot & \cdot & \times & \times & \times \\
\times & \times & \cdot & \cdot & \cdot \\
\times & \times & \cdot & \cdot & \cdot \\
\times & \times & \cdot & \cdot & \cdot
\end{pmatrix}
$$

因果时再与下三角取交集。由于打包顺序和序列内顺序一致，用全局位置比较 $j \le i$ 就够了。

## 形状

| 量 | 形状 |
| --- | --- |
| `q`、`k`、`v`、输出 | `(T, H, D)` |
| `cu_seqlens` | `(N + 1,)` |
| 序列编号 `seg` | `(T,)` |
| 掩码 | `(T, T)` |
| 分数 | `(H, T, T)` |

（真正的 varlen kernel 不会构造 $T \times T$ 矩阵，而是只计算对角块；本题用完整掩码表达同样的语义。）

## 用到的 PyTorch 操作

- `torch.searchsorted(cu_seqlens, pos, right=True) - 1`：对每个位置找到所属序列。`right=True` 保证长度为 $0$ 的序列（相邻边界相等）被正确跳过。
- 广播比较 `seg[:, None] == seg[None, :]` 得到 $T \times T$ 的布尔矩阵。
- `torch.einsum('ihd,jhd->hij', q, k)`：每个头内 token 两两点积；`'hij,jhd->ihd'` 做加权求和。
- `masked_fill(~mask, float('-inf'))`、`torch.softmax`。

## 常见坑

- **只加了因果掩码**：相邻序列会互相看到，打包就失去了意义。
- **`searchsorted` 用了默认的 `right=False`**：落在边界上的 token 会被分到前一个序列；有空序列时编号也会错。
- **把空序列当成错误**：长度为 $0$ 的序列合法，只是不包含任何 token。
- **einsum 的下标写错**：输入是 `(T, H, D)`，头维在中间。
- 缩放因子 $1/\sqrt{D}$。

## 延伸

- **省了多少**：长度 $[100, 2000, 300]$ 补齐到 $2000$ 后，注意力要算 $3 \times 2000^2 = 1.2 \times 10^7$ 个分数；打包后只算块内，共 $100^2 + 2000^2 + 300^2 \approx 4.1 \times 10^6$ 个。$\sum_j \ell_j^2$ 总是小于 $(\sum_j \ell_j)^2$，序列越多、长短越悬殊，差距越大。
- **位置编码要按序列重置**：打包后每个 token 的全局位置是 $0, 1, \dots, T-1$，但 RoPE 等位置编码需要的是它在**自己所属序列里**的位置，所以要用每条序列各自从 $0$ 开始的 position id（见「位置编码」专题）。
- **损失要在边界处屏蔽**：语言模型的目标是预测下一个 token。一条序列的最后一个 token 不应该去预测下一条序列的第一个 token，这些位置要从损失里排除。
- **不隔离的做法**：有些预训练流程只是把文档拼接起来、中间放一个结束符，只加因果掩码，让后面的文档也能看到前面的文档。块对角掩码更干净，代价是需要 varlen 形式的 kernel 或块掩码的支持。
- **kernel 的接口**：FlashAttention 的 `flash_attn_varlen_func` 除了 `cu_seqlens`，还要传最长序列的长度（`max_seqlen`），用于调度。PyTorch 的 FlexAttention 可以用一个掩码函数表达这类「文档掩码」。
- **推理中**：连续批处理把多个请求的 prompt 打包在一起做预填充，同样需要这种接口（见「推理服务」专题）。
- **负载不均**：计算量近似正比于 $\sum_j \ell_j^2$，而不是总 token 数。数据并行时，各卡 token 数相同但长序列多少不同，速度会有差别。

## 面试追问

1. **为什么要打包，而不是 padding？**
   padding 在补齐出来的位置上浪费计算和显存，长短悬殊时浪费很大。打包把所有序列首尾相接，不补任何东西，再用 `cu_seqlens` 标出边界，只算真实的 token。
2. **打包后只加因果掩码会发生什么？**
   后面序列的 token 可以看到前面序列的 token，互相泄漏信息，既污染了训练，推理时又不会有这样的上下文，造成训练和推理不一致。必须再加块对角掩码，让每个 token 只看同一条序列内的 token。
3. **`cu_seqlens` 是什么？怎么得到每个 token 所属的序列？**
   长度的前缀和，第 $j$ 条序列占 $[c_j, c_{j+1})$。对每个位置 $t$，用二分查找（`searchsorted`，`right=True`）找出最大的 $j$ 满足 $c_j \le t$。
4. **打包训练除了掩码，还要注意什么？**
   位置编码要在每条序列内重新从 $0$ 开始；序列边界处的预测不能计入损失。
5. **长度为 $0$ 或 $1$ 的序列会怎样？**
   长度 $0$ 的序列不包含 token，只在 `cu_seqlens` 里留下相邻相等的两项，不影响结果。长度 $1$ 的序列只能看见自己，输出就是它自己的 $v$。
