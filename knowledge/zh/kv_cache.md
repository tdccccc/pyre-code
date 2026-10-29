## 背景

语言模型逐个生成 token。生成第 $t$ 个 token 时，新 token 的查询 $q_t$ 要和**所有**历史 token 的 $k_1, \dots, k_t$、$v_1, \dots, v_t$ 做注意力。而历史 token 的 K、V 在之前的步骤里已经算过，并且不会改变（因果掩码保证前面的 token 看不到后面的）。

**KV 缓存**把算过的 K、V 存下来，每一步只为新 token 计算投影，然后拼到缓存后面。生成分两个阶段：

- **预填充**（prefill）：一次性处理整个提示词，$S_{\text{new}} > 1$，填满初始缓存
- **解码**（decode）：每步只输入 $1$ 个新 token，$S_{\text{new}} = 1$

## 核心公式

缓存里已有 $S_{\text{past}}$ 个位置，本次输入 $S_{\text{new}}$ 个：

$$
K_{\text{all}} = \bigl[\,K_{\text{cache}} ;\ K_{\text{new}}\,\bigr],
\qquad
V_{\text{all}} = \bigl[\,V_{\text{cache}} ;\ V_{\text{new}}\,\bigr]
$$

（$[\,\cdot\, ;\, \cdot\,]$ 表示沿序列维拼接），共 $S_{\text{total}} = S_{\text{past}} + S_{\text{new}}$ 个位置，然后

$$
\text{out} = \operatorname{softmax}\!\left(\frac{Q_{\text{new}} K_{\text{all}}^{\top}}{\sqrt{d_k}} + M\right) V_{\text{all}}
$$

**带偏移的因果掩码**：本次第 $i$ 个查询（$i = 0, \dots, S_{\text{new}} - 1$）在整个序列中的位置是 $S_{\text{past}} + i$，它能看到的键 $j$ 满足

$$
j \le S_{\text{past}} + i
$$

需要屏蔽的是 $j > S_{\text{past}} + i$，即形状为 $S_{\text{new}} \times S_{\text{total}}$ 的矩阵中，主对角线向右偏移 $S_{\text{past}} + 1$ 之后的上三角部分。解码时 $S_{\text{new}} = 1$，唯一的查询就是最新位置，所有键都可见，不需要掩码。

**省了多少计算**：不用缓存时，第 $t$ 步要重新投影全部 $t$ 个 token，生成 $T$ 个 token 共需 $\sum_{t=1}^{T} t = O(T^2)$ 次投影；使用缓存只需 $O(T)$ 次。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(B, S_new, d_model)` |
| 新的 $Q$、$K$、$V$（拆头后） | `(B, H, S_new, d_k)` |
| 缓存及拼接后的 $K$、$V$ | `(B, H, S_total, d_k)` |
| 分数、掩码 | `(B, H, S_new, S_total)`、`(S_new, S_total)` |
| 输出 | `(B, S_new, d_model)` |

## 用到的 PyTorch 操作

- `torch.cat([cache_k, k], dim=2)`：沿序列维拼接（拆头后的形状里序列在第 2 维）。
- `torch.triu(ones(S_new, S_total), diagonal=S_past + 1)`：`diagonal` 控制上三角从哪条对角线开始。
- 拆头 / 合头同多头注意力。

## 常见坑

- **拼接维度错**：`(B, H, S, d_k)` 的序列维是 `dim=2`，不是 `dim=1`。
- **掩码没有偏移**：直接用 `diagonal=1` 的方阵只适用于没有缓存的情况；有缓存时，新查询还要能看到全部历史位置。
- **返回的缓存不完整**：返回的应该是拼接**之后**的 K、V，下一步才能继续用。
- **缓存了投影前的输入**：缓存的是投影并拆头之后的 K、V，否则每步还得重新投影。
- 测试会检查：先预填充再逐个解码，结果必须和一次性完整前向完全一致。
