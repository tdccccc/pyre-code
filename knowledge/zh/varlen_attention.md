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
