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
