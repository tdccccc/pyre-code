## 背景

推理时的 KV 缓存是长上下文的主要显存开销。几种注意力变体每个 token、每层要缓存的元素数（$H$ 个头，每头 $d_h$ 维）：

| 方法 | 每 token 每层缓存 |
| --- | --- |
| MHA | $2\, H\, d_h$ |
| GQA（$H_{kv}$ 个 KV 头） | $2\, H_{kv}\, d_h$ |
| **MLA** | $d_c$（一个潜在向量） |

**MLA**（Multi-head Latent Attention，DeepSeek-V2/V3）不缓存 K、V 本身，而是缓存一个**低维潜在向量** $c$，需要时再从它「解压」出 K 和 V。DeepSeek-V2 中 $H = 128$、$d_h = 128$，MHA 每层要缓存 $32768$ 个数，而 MLA 的 $d_c = 512$，缩小了几十倍。

## 核心公式

**压缩**（down-projection）：

$$
c = X\, W_{\text{dkv}},\qquad W_{\text{dkv}} \in \mathbb{R}^{D \times d_c}
$$

**解压**（up-projection），K、V 共享同一个潜在向量：

$$
K = c\, W_{\text{uk}},\qquad V = c\, W_{\text{uv}},\qquad W_{\text{uk}}, W_{\text{uv}} \in \mathbb{R}^{d_c \times H d_h}
$$

**查询**：$Q = X\, W_q$。然后拆成 $H$ 个头，做普通的缩放点积注意力。

**低秩**：$K = X W_{\text{dkv}} W_{\text{uk}}$，这个 $D \times H d_h$ 的投影矩阵的秩不超过 $d_c$，这就是「低秩压缩」的含义。

**吸收技巧**（推理时的关键）：第 $h$ 个头的注意力分数

$$
q_h^{\top} k_h = q_h^{\top} \bigl(W_{\text{uk}}^{(h)}\bigr)^{\top} c = \Bigl(W_{\text{uk}}^{(h)}\, q_h\Bigr)^{\top} c
$$

可以把 $W_{\text{uk}}$ 并入查询一侧，直接和缓存的 $c$ 算点积，根本不需要显式解压出 K。$W_{\text{uv}}$ 同理可以并入输出投影。（实际 DeepSeek 的 MLA 还有一条单独的 RoPE 分支，本题不涉及。）

## 形状

| 量 | 形状 |
| --- | --- |
| `X` | `(B, S, D)` |
| 潜在向量 $c$ | `(B, S, kv_rank)` |
| $Q$、$K$、$V$ | `(B, S, H·d_h)` → 拆头为 `(B, H, S, d_h)` |
| 注意力分数 | `(B, H, S, S)` |
| 输出 | `(B, S, H·d_h)` |

$d_h$ 可以从 `W_q.shape[1] // num_heads` 得到。

## 用到的 PyTorch 操作

- `@` 矩阵乘。
- `t.view(B, S, H, d_h).transpose(1, 2)` 拆头；`transpose(1, 2).reshape(B, S, -1)` 合头。
- `torch.softmax`。

## 常见坑

- **K、V 用了不同的潜在向量**：两者都从同一个 $c$ 解压。
- **查询也经过压缩**：本题的 $Q$ 直接由 $X$ 投影。（DeepSeek-V2 也会压缩查询，但那是为了省训练时的激活，不影响 KV 缓存。）
- **拆头的 `view` 顺序**：先 `(B, S, H, d_h)` 再交换维度。
- **缩放因子**：$\sqrt{d_h}$，每个头的维度。
