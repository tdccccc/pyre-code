## 背景

想让模型更强通常要增加参数，但参数越多、每个 token 的计算量也越大。**混合专家**（Mixture of Experts，MoE）打破了这个绑定：把一个 FFN 换成 $E$ 个并列的「专家」FFN，再加一个**路由器**，每个 token 只送给其中 $k$ 个专家（通常 $k = 1$ 或 $2$）。

- 总参数量 ≈ $E$ 个 FFN
- 每个 token 的计算量 ≈ $k$ 个 FFN

Mixtral 8×7B 共有约 47B 参数，但每个 token 只激活约 13B。DeepSeek-V3、Qwen-MoE 等模型都使用 MoE。

## 核心公式

**路由**：路由器是一个线性层，为每个专家打分：

$$
\ell = x\, W_r \in \mathbb{R}^{E}
$$

选出得分最高的 $k$ 个专家 $\mathcal{T} = \operatorname{topk}(\ell)$，只在这 $k$ 个分数上做 softmax 得到门控权重：

$$
g_i = \frac{e^{\ell_i}}{\sum_{j \in \mathcal{T}} e^{\ell_j}},\qquad i \in \mathcal{T}
$$

这和「先对全部 $E$ 个分数做 softmax，再把选中的 $k$ 个概率重新归一化」的结果完全一样，因为分母中的公共因子会约掉。

**输出**：被选中专家输出的加权和：

$$
y = \sum_{i \in \mathcal{T}} g_i \cdot \operatorname{Expert}_i(x),
\qquad
\operatorname{Expert}_i(x) = W_2^{(i)}\, \operatorname{ReLU}\bigl(W_1^{(i)} x + b_1^{(i)}\bigr) + b_2^{(i)}
$$

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(B, S, d_model)`，展平为 `(B·S, d_model)` 个 token |
| 路由分数 | `(B·S, E)` |
| top-k 的分数、下标、门控 | `(B·S, k)` |
| 输出 | `(B, S, d_model)` |

## 实现思路

逐个 token 调用专家太慢。更好的做法是**按专家分组**：对每个专家 $e$，找出所有选中它的 token（布尔掩码），把它们堆在一起一次送进专家，再把结果乘以对应的门控、加回这些 token 的输出位置。

## 用到的 PyTorch 操作

- `nn.ModuleList([...])`：存放多个专家，子模块才会被注册（普通 Python 列表不会）。
- `logits.topk(k, dim=-1)`、`torch.softmax(vals, dim=-1)`。
- 布尔掩码 `mask = top_idx[:, j] == e`，`x[mask]` 取出这些 token。
- `x.reshape(-1, D)` 展平，最后 `reshape` 回原形状。

## 常见坑

- **对全部专家做 softmax 后直接取 top-k 概率**：没有重新归一化，权重之和小于 $1$，与本题要求不同。
- **用 Python 列表存专家**：参数不会被注册，优化器找不到，测试也会检查 `self.experts`。
- **没有展平 batch 和序列维**：路由是逐 token 的。
- **门控形状不匹配**：`(n, 1)` 的门控才能和 `(n, d_model)` 的专家输出广播相乘。
