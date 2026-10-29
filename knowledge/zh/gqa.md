## 背景

大模型推理时，每生成一个 token 都要把所有历史 token 的 K、V 缓存起来（KV cache）。缓存大小为

$$
\text{KV cache} = 2 \times L \times H_{kv} \times d_k \times S \times \text{每个数的字节数}
$$

（$L$ 层，$H_{kv}$ 个 KV 头，序列长 $S$，因子 2 表示 K 和 V 各一份）。长上下文、大 batch 时它比模型权重还大，而且每步都要从显存读一遍，成为推理瓶颈。

缩小它的办法是减少 KV 头数 $H_{kv}$：

| 方法 | KV 头数 | 说明 |
| --- | --- | --- |
| MHA（多头注意力） | $H_{kv} = H$ | 每个查询头有自己的 K/V |
| MQA（多查询注意力） | $H_{kv} = 1$ | 所有查询头共享一组 K/V，省得最多，但质量有损失 |
| **GQA（分组查询注意力）** | $1 < H_{kv} < H$ | 每 $g = H / H_{kv}$ 个查询头共享一组 K/V |

LLaMA 2/3、Mistral 都用 GQA，KV cache 缩小为 MHA 的 $H_{kv}/H$。

## 核心公式

查询头 $h$（$h = 0, \dots, H-1$）使用第 $\lfloor h / g \rfloor$ 个 KV 头：

$$
\text{head}_h = \operatorname{softmax}\!\left(\frac{Q_h K_{\lfloor h/g \rfloor}^{\top}}{\sqrt{d_k}}\right) V_{\lfloor h/g \rfloor},
\qquad g = \frac{H}{H_{kv}}
$$

例如 $H = 4$、$H_{kv} = 2$：查询头 $0, 1$ 用 KV 头 $0$；查询头 $2, 3$ 用 KV 头 $1$。

## 形状

| 量 | 形状 |
| --- | --- |
| $W_Q$ | `Linear(d_model, d_model)`，$d_{\text{model}} = H \cdot d_k$ |
| $W_K$、$W_V$ | `Linear(d_model, H_kv · d_k)` |
| 拆头后的 $Q$ | `(B, H, S, d_k)` |
| 拆头后的 $K$、$V$ | `(B, H_kv, S, d_k)` |
| 把 KV 头扩展后 | `(B, H, S, d_k)` |

扩展之后就是普通的多头注意力。

## 用到的 PyTorch 操作

- `t.repeat_interleave(g, dim=1)`：把每个头**连续**复制 $g$ 次：$[k_0, k_1] \to [k_0, k_0, k_1, k_1]$。
- 拆头、合头同多头注意力。

## 常见坑

- **用了 `repeat`**：`t.repeat(1, g, 1, 1)` 是整体平铺，$[k_0, k_1] \to [k_0, k_1, k_0, k_1]$，查询头和 KV 头的对应关系就错了。
- **$W_K$、$W_V$ 输出维度写成 $d_{\text{model}}$**：应为 $H_{kv} \cdot d_k$，测试会检查形状。
- **$d_k$ 算错**：$d_k = d_{\text{model}} / H$，按**查询**头数计算。
- 自检：$H_{kv} = H$ 时必须退化为标准 MHA。
