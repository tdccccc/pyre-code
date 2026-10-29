## 背景

注意力机制让序列中的每个位置去「查询」其他位置，按相关程度加权汇总信息。可以把它想成一次软性的字典查找：

- **Query（查询）** $q$：我想找什么
- **Key（键）** $k$：每个位置「我有什么」的标签
- **Value（值）** $v$：每个位置真正提供的内容

用 $q$ 和每个 $k$ 的相似度决定从每个 $v$ 取多少。

## 核心公式

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V
$$

拆成三步：

1. **打分**：$S = Q K^{\top} / \sqrt{d_k}$，$S_{ij} = q_i \cdot k_j / \sqrt{d_k}$ 表示第 $i$ 个查询和第 $j$ 个键的相似度（点积）。
2. **归一化**：$A = \operatorname{softmax}(S)$，对每一行（每个查询）做 softmax，$\sum_j A_{ij} = 1$。
3. **加权求和**：$O = A V$，即 $o_i = \sum_j A_{ij}\, v_j$。

**为什么除以 $\sqrt{d_k}$**：设 $q, k$ 的各分量独立、均值 $0$、方差 $1$，则

$$
\operatorname{Var}(q \cdot k) = \operatorname{Var}\!\left(\sum_{t=1}^{d_k} q_t k_t\right) = d_k
$$

维度越大，点积的数值越大，softmax 会变得极其尖锐（几乎变成 one-hot），梯度接近 $0$。除以标准差 $\sqrt{d_k}$ 把方差拉回 $1$。

## 形状

| 量 | 形状 |
| --- | --- |
| $Q$ | `(B, S_q, D)` |
| $K$ | `(B, S_k, D)` |
| $V$ | `(B, S_k, D_v)` |
| $Q K^{\top}$、$A$ | `(B, S_q, S_k)` |
| 输出 $A V$ | `(B, S_q, D_v)` |

$S_q$ 和 $S_k$ 可以不相等（交叉注意力：查询来自一个序列，键值来自另一个）。softmax 沿最后一维 $S_k$ 做，也就是每个查询在所有键上归一化。

## 用到的 PyTorch 操作

- `K.transpose(1, 2)` 或 `K.transpose(-2, -1)`：交换最后两维，得到 `(B, D, S_k)`。
- `torch.bmm(a, b)`：批量矩阵乘，`(B, n, m) @ (B, m, p) → (B, n, p)`；`@` / `torch.matmul` 也可以。
- `torch.softmax(x, dim=-1)`。
- `K.size(-1)` 取 $d_k$，`math.sqrt` 开方。

## 常见坑

- **转置了错误的维度**：`K.T` 对三维张量会把所有维度倒过来，变成 `(D, S_k, B)`。要只交换最后两维。
- **softmax 方向错了**：沿 `dim=-1`（键的方向）归一化。沿 `dim=1` 归一化，形状照样对得上，但结果是错的。
- **缩放因子写错**：除以 $\sqrt{d_k}$，不是 $d_k$；$d_k$ 是 Q/K 的特征维，不是序列长度。
