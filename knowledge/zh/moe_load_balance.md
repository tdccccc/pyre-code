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
