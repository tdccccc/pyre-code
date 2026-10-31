## 背景

一次计算的耗时由两个上限中较慢的那个决定：**算力**（每秒能做多少 FLOPs）和**显存带宽**（每秒能读写多少字节）。**Roofline 模型**用一个量来判断瓶颈在哪：**算术强度**（arithmetic intensity），即每读一个字节做多少次运算。

LLM 解码时每生成一个 token 都要把全部权重和 KV 缓存读一遍，却只做很少的运算，所以几乎总是**受带宽限制**。理解这一点，就能理解 GQA、更大的 batch、连续批处理、量化等推理优化为什么有效。

## 核心公式

**Roofline**：

$$
t = \max\!\left(\frac{\text{FLOPs}}{\text{峰值 FLOP/s}},\ \frac{\text{Bytes}}{\text{带宽}}\right)
$$

**算术强度**与**拐点**（ridge point）：

$$
I = \frac{\text{FLOPs}}{\text{Bytes}},
\qquad
I_{\text{ridge}} = \frac{\text{峰值 FLOP/s}}{\text{带宽}}
$$

$I < I_{\text{ridge}}$ 时受带宽限制（memory bound），反之受算力限制（compute bound）。A100 的拐点约为 $312 \times 10^{12} / 2 \times 10^{12} = 156$ FLOP/字节。

**一个解码步**（$\Psi$ 参数，batch 为 $b$，bf16 每个数 2 字节）：

$$
\text{FLOPs} = 2\, \Psi\, b,
\qquad
\text{Bytes} = 2\Psi + \text{KV}
$$

权重对整个 batch 只读一次，KV 缓存每个序列各读一遍：

$$
\text{KV} = 2 \cdot L \cdot b \cdot s \cdot H_{kv} \cdot d_h \cdot 2 \text{ 字节}
$$

**为什么 batch 有用**：先忽略 KV，

$$
I \approx \frac{2\Psi b}{2\Psi} = b
$$

batch 为 $1$ 时 $I \approx 1$，远低于 $156$，GPU 算力只用了不到 $1\%$。把 batch 加大，一次权重读取被摊给更多 token，吞吐近似线性增长，直到进入 compute bound。

**为什么 KV 缓存会拖后腿**：KV 字节数也正比于 $b$，当 $b \to \infty$：

$$
I = \frac{2\Psi b}{2\Psi + b \cdot \text{KV}_{\text{单序列}}} \ \to\ \frac{2\Psi}{\text{KV}_{\text{单序列}}}
$$

长上下文时这个上限可能低于拐点，再加 batch 也到不了 compute bound。GQA 减少 $H_{kv}$，直接缩小 $\text{KV}_{\text{单序列}}$，抬高了这个上限。

**MFU**：

$$
\text{MFU} = \frac{\text{FLOPs}}{t \cdot \text{峰值 FLOP/s}} = \min\!\left(1,\ \frac{I}{I_{\text{ridge}}}\right)
$$

## 常见坑

- **权重按 batch 重复读取**：整个 batch 只读一次权重，这正是 batching 的意义。
- **KV 缓存漏了因子 2**：K 和 V 各一份。
- **`head_dim` 用 KV 头数算**：`head_dim = d_model // n_heads`，按查询头数计算；KV 头数只影响缓存里有几个头。
- **边界判断**：$I < I_{\text{ridge}}$ 为 `memory`，否则为 `compute`。
- `kv_cache_bytes`、`flops`、`bytes` 返回 `int`，其余为 `float`。

## 延伸

**拐点随硬件上升**。拐点是峰值算力除以带宽。A100 约 $156$ FLOP/字节，H100 SXM 约 $295$，算力涨得比带宽快，意味着同样的 decode 在新硬件上离「用满算力」更远，带宽的优化更重要。

**batch 1 的速度上界**。忽略 KV 时，每步至少要把权重读一遍：

$$
\text{token/s} \le \frac{\text{带宽}}{\text{权重字节数}}
$$

7B 模型 bf16 权重约 $13.5$ GB，A100 的 $2$ TB/s 下约 $148$ token/s。权重量化到 int8，字节数减半，上界约翻倍（见「量化与高效微调」专题）。这解释了为什么权重量化对 decode 特别有效，而对 compute bound 的 prefill 帮助有限。

**KV 缓存的账**。每个缓存 token 的字节数：

$$
2 \cdot L \cdot H_{kv} \cdot d_h \cdot \text{bytes}
$$

7B 的 MHA 配置（$L=32$，$H_{kv}=32$，$d_h=128$，bf16）是 $0.5$ MiB/token，一个 2048 token 的序列就是 $1$ GiB；换成 8 个 KV 头的 GQA，缩小 4 倍。batch 的上限常常不是算力，而是容量：

$$
b_{\max} \approx \frac{\text{显存} - \text{权重}}{\text{单序列 KV}}
$$

分页管理 KV 缓存、前缀共享等技术正是为了在这个约束下放进更多序列（见「推理服务」专题）。

**注意力自己的算术强度不随 batch 变化**。decode 的注意力里，缓存里的每个 K、V 元素被读一次，被同组的 $g = H / H_{kv}$ 个查询头各用一次乘加，bf16 下算术强度约为 $g$（MHA 是 $1$）。batch 变大只是读更多序列自己的 KV，每个序列的强度不变，所以 KV 部分始终压低整体的算术强度，这就是前文中 $b \to \infty$ 时有上限的原因。GQA 和 MQA 把 $g$ 提高到 $4$、$8$ 或更多，直接改善这部分（见「注意力」专题）。

**prefill 与 decode**。prefill 一次处理整段 prompt，$T$ 个 token 摊薄一次权重读取，算术强度约 $T$，通常是 compute bound，对应首 token 时间（TTFT）；decode 每步每序列一个 token，通常是 memory bound，对应逐 token 延迟。把二者混在同一个 batch 里，或者把长 prompt 切块与 decode 一起调度，是推理服务里的常见做法（见「推理服务」专题）。

**投机解码的角度**。decode 的算力大部分闲置，那么一次读权重就多验证几个候选 token，把闲置算力换成吞吐（见「解码与采样」专题）。

**本题简化的地方**。忽略注意力本身的 FLOPs、激活、多卡之间的通信（张量并行每层有 all-reduce）；用 $\max$ 表示计算和访存完美重叠；实际 kernel 很难达到带宽峰值。所以结果是上界。MoE 的情形也不同：batch 大时，不同 token 会命中几乎所有专家，需要读取的权重趋近全部专家，而不是只有激活的那几个。

## 面试追问

1. **为什么 decode 是 memory bound？**
   每生成一个 token 都要把全部权重和该序列的 KV 缓存读一遍，只做约 $2\Psi$ 次运算。batch 为 1 时算术强度约 $1$，远低于拐点（A100 约 $156$）。算力闲着，时间由读字节的速度决定。
2. **增大 batch 对 decode 有什么帮助，有上限吗？**
   权重只读一次却服务整个 batch，所以算术强度近似随 batch 线性增长，吞吐也近似线性增长。但 KV 缓存每个序列各读一遍，使强度趋于上限 $2\Psi / \text{KV}_{\text{单序列}}$；此外 KV 的显存容量也限制了能同时放下的序列数。
3. **GQA 为什么能加速 decode？**
   KV 头数减少，每个 token 缓存的字节数按比例缩小。读取量更小，算术强度更高，同样显存能放更多序列，batch 可以更大。
4. **7B 模型在 A100 上 batch 1 的 decode 速度上界是多少？**
   权重 bf16 约 $13.5$ GB，带宽 $2$ TB/s，读一遍约 $6.7$ ms，上界约 $148$ token/s；加上 2k 上下文的 MHA KV（$1$ GiB）约 $7.3$ ms，约 $137$ token/s。
5. **prefill 和 decode 的瓶颈有什么不同，优化方向呢？**
   prefill 是 compute bound，提升靠更高的算力利用率（kernel 效率、并行）；decode 是 memory bound，靠少搬字节（量化、GQA、KV 管理）和多摊薄（连续批处理、投机解码）。
