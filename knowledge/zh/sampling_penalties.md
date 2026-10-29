## 背景

语言模型容易陷入重复：同一个词、同一句话反复出现。推理引擎在采样前会对 logits 做一系列处理（logits processor），常见的有：

- **重复惩罚**（repetition penalty，CTRL 论文提出）：出现过的 token 一律降低概率
- **频率惩罚**、**存在惩罚**（OpenAI API 中的 `frequency_penalty`、`presence_penalty`）：按出现次数或是否出现扣分
- **温度**
- **min-p 过滤**：去掉相对最可能 token 而言概率太低的 token

## 核心公式

**重复惩罚**（$\theta > 1$，每个出现过的不同 token 只做一次）：

$$
z_t \leftarrow
\begin{cases}
z_t / \theta, & z_t > 0 \\
z_t \cdot \theta, & z_t \le 0
\end{cases}
$$

为什么要分正负：目的是**降低** logit。正数除以 $\theta$ 变小；负数如果也除以 $\theta$ 会变得更接近 $0$，反而**升高**了，所以负数要乘以 $\theta$。

**频率与存在惩罚**（$c_t$ 为 token $t$ 已出现的次数）：

$$
z_t \leftarrow z_t - \alpha_{\text{freq}} \cdot c_t - \alpha_{\text{pres}} \cdot \mathbb{1}[c_t > 0]
$$

频率惩罚随次数累加，存在惩罚只要出现过就扣一次。

**温度**：$z \leftarrow z / T$，然后 $p = \operatorname{softmax}(z)$。

**min-p**：保留相对于最大概率不太小的 token，再重新归一化：

$$
\text{保留 } i \iff p_i \ge p_{\min} \cdot \max_j p_j
$$

和 top-p 相比，min-p 的阈值随模型的「信心」自动调整：最大概率很高（模型很确定）时阈值高，砍掉更多；分布平坦时阈值低，保留更多候选。

**顺序**：惩罚 → 温度 → softmax → min-p。惩罚作用于温度缩放**之前**的 logits。

## 用到的 Python / PyTorch 工具

- `collections.Counter(prev_tokens)`：统计每个 token 出现的次数，键就是去重后的 token。
- `logits.clone()`：先复制，避免修改调用方的张量。
- 用 token 列表做高级索引：`logits[ids]` 取出、`logits[ids] = vals` 写回。
- `torch.where(cond, a, b)`：按条件逐元素选择，适合做分正负的惩罚和 min-p 过滤。

## 常见坑

- **重复出现的 token 被重复惩罚**：重复惩罚对每个**不同**的 token 只做一次；次数只影响频率惩罚。
- **存在惩罚乘了次数**：存在惩罚与次数无关。
- **负 logit 也用除法**：会让该 token 概率上升。
- **先温度后惩罚**：顺序不同结果不同，测试会检查。
- **min-p 边界**：恰好等于阈值的保留，只去掉严格小于阈值的。
- **原地修改了输入**的 `logits`。
