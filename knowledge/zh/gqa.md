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

LLaMA 2 的 70B、LLaMA 3 全系列、Mistral 7B 都用 GQA，KV cache 缩小为 MHA 的 $H_{kv}/H$。

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

## 延伸

- **缓存的是扩展前的 K、V**：`repeat_interleave` 只是把写法对齐到 MHA，真正存进 KV cache 的是 $H_{kv}$ 份。高效的 kernel 也不会物理复制：同一组的 $g$ 个查询头直接读同一份 K、V，访存量不变，计算量与访存量之比提高了 $g$ 倍（见「性能估算」专题的推理 roofline）。
- **参数量**：设 $d_{\text{model}} = H d_k$，$W_Q$、$W_O$ 各 $d_{\text{model}}^2$，$W_K$、$W_V$ 各 $d_{\text{model}} \cdot H_{kv} d_k$，总计

$$
2\, d_{\text{model}}^2 \Bigl(1 + \frac{H_{kv}}{H}\Bigr)
$$

  比 MHA 的 $4\, d_{\text{model}}^2$ 略少，但省下的参数并不是目的，缓存才是。
- **从 MHA 转换**：GQA 的论文提出，可以对同一组内各头的 K、V 投影权重取平均，作为 GQA 的初始化，然后只用原预训练计算量的一小部分继续训练，质量就能基本恢复。
- **和张量并行配合**：张量并行按头切分，若 $H_{kv}$ 比并行度还小，就需要在多张卡上复制 KV 头（见「分布式训练」专题）。
- **实际取值**：LLaMA 3 8B 有 $32$ 个查询头、$8$ 个 KV 头（$g = 4$），常见的配置里 $H_{kv}$ 取 $8$ 左右。

## 面试追问

1. **GQA 相比 MHA 省了什么？**
   主要省 KV cache：每个 token、每层的缓存从 $2 H d_k$ 降到 $2 H_{kv} d_k$，是 MHA 的 $H_{kv} / H$。解码时每步要读的数据量随之减少，省下的显存也可以用来放更大的 batch 或更长的上下文。注意力分数的计算量不变。
2. **GQA 与 MHA、MQA 是什么关系？**
   GQA 是两者的推广：$H_{kv} = H$ 时就是 MHA，$H_{kv} = 1$ 时就是 MQA。分组数是一个可以调的旋钮，在质量和缓存之间折中。
3. **为什么 MQA 的质量会下降？**
   所有查询头共享同一组 K、V，各头只能从同一个键值表示里取信息，表达力受限。分成几组后，不同组可以有不同的 K、V，质量接近 MHA。（MQA 在大规模训练时有时还会不稳定。）
4. **用 `repeat_interleave` 扩展 KV 头，缓存里存的是扩展前还是扩展后的？**
   扩展前。扩展是为了让实现和 MHA 对齐；缓存扩展后的结果会把省下的显存全部还回去。
5. **为什么解码阶段 GQA 比 MHA 更快？**
   解码时每步只处理一个新 token，计算量很小，时间主要花在把 KV cache 从显存读出来。GQA 的缓存更小，读得更快。预填充阶段是计算受限的，GQA 的收益小得多。
