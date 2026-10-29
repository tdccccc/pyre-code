## 背景

分页 KV 缓存把每个序列的 K、V 分散存放在不连续的物理页里（见「KV 块分配器」一题）。注意力计算时就不能直接拿一段连续的张量，而要先通过**块表**（block table）找到这个序列的各页，再按顺序取出。这就是 vLLM 的 **PagedAttention**。

它和操作系统的虚拟内存一一对应：

| 操作系统 | PagedAttention |
| --- | --- |
| 虚拟页 | 逻辑块（序列中的第几块） |
| 物理页帧 | 物理页（KV 缓存池中的第几页） |
| 页表 | 块表 |
| 页内偏移 | 块内第几个 token |

## 核心公式

**地址转换**：每页存 $P$ 个 token（`block_size`），序列中第 $t$ 个 token 位于

$$
\text{物理页} = \text{block\_table}\bigl[\lfloor t / P \rfloor\bigr],
\qquad
\text{页内偏移} = t \bmod P
$$

**取出 K/V**：按块表顺序把各页取出、首尾相接，得到一段连续的 $K$、$V$，再截取前 $n$ 个（$n$ = `context_len`）有效 token。

**注意力**（单步解码，每个序列一个查询）：

$$
o = \operatorname{softmax}\!\left(\frac{q\, K_{1:n}^{\top}}{\sqrt{D}}\right) V_{1:n}
$$

**浪费**：每个序列只有最后一页可能没填满，浪费不超过 $P - 1$ 个 token 的空间。

## 形状

| 量 | 形状 |
| --- | --- |
| `Q` | `(B, 1, D)` |
| `k_pages`、`v_pages` | `(num_pages, P, D)` |
| `block_table` | `(B, max_blocks)` |
| `k_pages[block_table[b]]` | `(max_blocks, P, D)` |
| `reshape(-1, D)` 后 | `(max_blocks · P, D)` |
| 截取后 | `(n, D)` |
| 输出 | `(B, 1, D)` |

## 用到的 PyTorch 操作

- **用整数张量索引**：`k_pages[idx]`，`idx` 是一维整数张量，结果按 `idx` 的顺序取出各页。
- `t.reshape(-1, D)`：把 `(页数, P, D)` 展平成按 token 排列的 `(页数 · P, D)`，块内顺序保持不变。
- 切片 `[:context_len]`、`torch.softmax`、`@`。

## 常见坑

- **没有截取到 `context_len`**：最后一页未写满的位置是无效数据，参与 softmax 会污染结果。
- **展平维度错**：先 `(页, token, D)` → `(页·token, D)`，顺序是第 0 页的全部 token、再第 1 页……
- **所有序列共用一个块表行**：每个序列 $b$ 用自己的 `block_table[b]`。
- **缩放因子**：除以 $\sqrt{D}$。
