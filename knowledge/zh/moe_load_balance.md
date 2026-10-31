## 背景

MoE 训练有一个常见的退化：路由器一开始稍微偏爱某几个专家，这些专家因为收到更多 token 而训练得更好，于是路由器更加偏爱它们……最后几乎所有 token 都挤到少数专家上（**路由坍缩**），其余专家闲置，MoE 的容量被白白浪费；在专家并行中还会让个别 GPU 过载。

**负载均衡辅助损失**（Switch Transformer 提出）在主损失之外加一项，惩罚不均匀的分配。

## 核心公式

$N$ 个 token，$E$ 个专家，路由概率 $p_{n,i} = \operatorname{softmax}(\ell_n)_i$。

**实际分配比例**（每个 token 按 top-1 分给一个专家）：

$$
f_i = \frac{1}{N} \sum_{n=1}^{N} \mathbb{1}\bigl[\arg\max_j \ell_{n,j} = i\bigr]
$$

**平均路由概率**：

$$
P_i = \frac{1}{N} \sum_{n=1}^{N} p_{n,i}
$$

**辅助损失**：

$$
L_{\text{aux}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

**为什么是这个形式**：$f_i$ 来自 argmax，不可导；$P_i$ 可导。损失对 $P_i$ 的梯度是

$$
\frac{\partial L_{\text{aux}}}{\partial P_i} = E \cdot f_i
$$

收到 token 越多的专家（$f_i$ 越大），它的路由概率被往下压得越狠。梯度经过 $P_i$ 流回路由器的 logits。

**取值**：

- 完全均匀：$f_i = P_i = 1/E$，$L_{\text{aux}} = E \cdot E \cdot \frac{1}{E^2} = 1$
- 完全坍缩到专家 $j$：$f_j = 1$，$P_j \approx 1$，$L_{\text{aux}} \approx E$

乘以 $E$ 让均匀时的损失恒为 $1$，与专家数无关。实际训练中这一项还要乘一个小系数（如 $0.01$）再加到主损失上。

## 形状

| 量 | 形状 |
| --- | --- |
| `router_logits`、$p$ | `(N, E)` |
| argmax 分配 | `(N,)` |
| $f$、$P$ | `(E,)` |
| 输出 | 标量 |

## 用到的 PyTorch 操作

- `logits.argmax(dim=-1)`：每个 token 的 top-1 专家。
- `(assign == e).float().mean()`：分给专家 $e$ 的比例；也可以用 `torch.bincount(assign, minlength=E) / N`。
- `torch.softmax(logits, dim=-1).mean(dim=0)`：对 token 维求平均。

## 常见坑

- **$f$ 用 softmax 概率计算**：$f$ 是硬分配的计数比例，来自 argmax。
- **漏乘 $E$**：均匀时损失应为 $1$，测试会检查。
- **$P$ 沿错误的维度平均**：对 token 维（`dim=0`）平均，得到每个专家一个值。
- 梯度只经过 $P$，$f$ 不需要（也不能）求导。

## 延伸

- **损失的来源与系数**：Switch Transformer 提出这一形式，系数 $\alpha$ 取 $10^{-2}$ 量级，即 $L = L_{\text{main}} + \alpha L_{\text{aux}}$。$\alpha$ 太小压不住坍缩，太大则路由被迫平均，损害模型质量。
- **top-$k$ 的推广**：本题的 $f_i$ 只按 top-1 统计。$k > 1$ 时，常见做法是统计每个专家被选进 top-$k$ 的 token 比例，再按需要缩放，具体归一化各模型略有差别。
- **统计范围**：$f$ 和 $P$ 可以按单个序列、单个 micro-batch 或整个全局 batch 来统计，范围越小约束越严，对模型的干扰越大。
- **容量限制**：Switch Transformer 给每个专家设了容量上限 $\text{capacity} = \text{capacity\_factor} \cdot \frac{N}{E}$，超出的 token 会被丢弃，直接走残差。辅助损失的目的之一就是减少丢弃。
- **路由 z-loss**：ST-MoE 另外加了一项惩罚 router logits 的 $\operatorname{logsumexp}$ 过大，用来稳定训练，它与负载均衡是两件事：

$$
L_z = \frac{1}{N} \sum_{n=1}^{N} \Bigl( \log \sum_{i} e^{\ell_{n,i}} \Bigr)^2
$$

- **无辅助损失的均衡**：辅助损失的梯度会和主任务的梯度打架。DeepSeek-V3 改成给每个专家维护一个偏置 $b_i$，**只在选 top-$k$ 时**把它加到分数上，门控权重仍用原始分数：

$$
\mathcal{T} = \operatorname{topk}_i \,(s_i + b_i),
\qquad
g_i \propto s_i \ (i \in \mathcal{T})
$$

  每个训练步之后检查各专家的负载：过载的专家把 $b_i$ 减去一个固定步长，欠载的加上一个步长。偏置不参与反向传播，所以不会干扰主损失的梯度。DeepSeek-V3 另外还保留了一个系数极小的序列级辅助损失，防止单个序列内部过度不均。
- **专家并行**：负载不均在多卡上会变成实际的卡间等待，因为最慢的那张卡决定整步时间（见「分布式训练」专题）。

## 面试追问

1. **为什么 $f_i$ 不可导，损失还能训练路由器？**
   梯度走 $P_i$ 这一条路：$\frac{\partial L_{\text{aux}}}{\partial P_i} = E f_i$，$f_i$ 当作常数。收到 token 多的专家，它的平均路由概率被压得更狠，路由器因此把 token 往别处分。
2. **为什么要乘 $E$？**
   完全均匀时 $f_i = P_i = 1/E$，$\sum_i f_i P_i = 1/E$，乘 $E$ 后恒等于 $1$，和专家数无关，系数 $\alpha$ 就不必随专家数重调。完全坍缩时约为 $E$。
3. **为什么 $f$ 和 $P$ 两项都要，只用 $f$ 或只用 $P$ 不行吗？**
   只用 $f$ 没有梯度；只用 $P$ 则只鼓励概率平均，但 argmax 分配仍可能不均。两项相乘，既能反映真实的分配，又能让梯度传回路由器。
4. **辅助损失和无辅助损失的偏置方法各有什么优缺点？**
   辅助损失实现简单，但系数是在均衡和模型质量之间折中，梯度会干扰主任务；偏置方法不引入额外梯度，但要在训练循环里额外维护偏置、设置步长，并且是事后调整，不是可微的目标。
5. **容量因子（capacity factor）是什么？**
   每个专家一次最多接收 $\text{capacity\_factor} \cdot N / E$ 个 token，多出来的被丢弃。因子大则丢得少但计算和显存浪费多，因子小则相反。
