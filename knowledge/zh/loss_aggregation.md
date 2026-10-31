## 背景

PPO、GRPO 以及它们的各种变体，都是先算出每个 token 的损失 $\ell_{i,t}$，得到一个 `(B, T)` 的矩阵，最后再归约成一个标量去反向传播。归约看起来只是「取个平均」，实际上它决定了**每个 token 在梯度里的权重**：同样的优势，长回答和短回答的 token 分到的权重可能差好几倍。

2025 年围绕 GRPO 的几篇工作（DAPO、Dr. GRPO）都把聚合方式当作要修的 bug 之一。现在框架把它做成一个配置项：verl 叫 `loss_agg_mode`，默认 `token-mean`；TRL 用 `loss_type` 区分，`"grpo"`、`"dr_grpo"`、`"dapo"` 各对应一种聚合，默认是 `"dapo"`。

## 核心公式

记 $m_{i,t}$ 为掩码，$n_i = \sum_t m_{i,t}$ 为回答 $i$ 的有效 token 数，$N = \sum_i n_i$ 为 batch 里的有效 token 总数。四种方式写成「每个有效 token 的权重 $w_{i,t}$」最直观，结果都是 $\sum_{i,t} w_{i,t}\,\ell_{i,t}$：

| 模式 | 结果 | 每个 token 的权重 $w_{i,t}$ |
| --- | --- | --- |
| `token-mean` | $\dfrac{\sum_{i,t} m\,\ell}{N}$ | $\dfrac{1}{N}$ |
| `seq-mean-token-mean` | $\dfrac{1}{B}\sum_i \dfrac{\sum_t m\,\ell}{n_i}$ | $\dfrac{1}{B\,n_i}$ |
| `seq-mean-token-sum` | $\dfrac{1}{B}\sum_i \sum_t m\,\ell$ | $\dfrac{1}{B}$ |
| `seq-mean-token-sum-norm` | $\dfrac{1}{B}\sum_i \dfrac{\sum_t m\,\ell}{L}$ | $\dfrac{1}{B\,L}$ |

$L$ 是一个与回答长度无关的常数（本题默认取 padding 后的宽度 $T$，实际训练中常取最大生成长度）。

看权重这一列就能看出区别：

- `seq-mean-token-mean` 是 DeepSeekMath 里 GRPO 的原始写法，权重和回答长度 $n_i$ 成反比。
- `token-mean` 让 batch 里每个 token 平等，但权重依赖这个 batch 的 token 总数 $N$。
- `seq-mean-token-sum-norm` 是 Dr. GRPO 的写法，权重是常数，既不看本回答多长，也不看这个 batch 有多少 token。

## 形状

- `token_loss`、`mask` 都是 `(B, T)`，输出是标量。
- `(token_loss * m).sum(dim=1)` 得到每个回答的损失之和 `(B,)`；`m.sum(dim=1)` 得到每个回答的长度 `(B,)`。四种模式都由这两个向量组合出来。

## 用到的 PyTorch 操作

- `mask.to(token_loss.dtype)`：bool 掩码转成浮点，才能相乘。
- `t.sum(dim=1)`、`t.sum()`、`t.mean()`：按行求和、整体求和、对行取平均。
- `raise ValueError(...)`：不认识的模式要报错，别默默落到某个默认分支。

## 常见坑

- **`token-mean` 除以了 `numel()`**：分母要数有效 token，padding 不算，否则 padding 越多损失越小。
- **`seq-mean-token-sum-norm` 除以每行自己的长度**：那就退化成了 `seq-mean-token-mean`。Dr. GRPO 的要点正是用常数替换 $n_i$。
- **`norm_len` 被忽略**：常数 $L$ 是可配置的，只有传 `None` 时才用 $T$。
- **按行求和后用 `sum` 而不是 `mean`**：结果会随 batch 大小线性变大，相当于偷偷改了学习率。
- **padding 位置参与计算**：先乘掩码再求和。padding 上可能是任意值（比如对数概率里的填充），不能假设它是 0。

## 延伸

- **长度偏差从哪来**：在 `seq-mean-token-mean` 下，同样的优势 $A$，短回答每个 token 拿到 $A/(B n_i)$，长回答被摊薄。$A > 0$ 时模型更偏爱短的正确回答；$A < 0$ 时，长的错误回答每个 token 挨的罚更轻。Dr. GRPO（*Understanding R1-Zero-Like Training*，2025）认为这会让错误回答越写越长，于是去掉 $1/n_i$，改成除以常数，同时还去掉了优势里的标准差归一化（见「组内基线：GRPO / Dr. GRPO / RLOO / REINFORCE++」一题）。
- **DAPO 为什么选 `token-mean`**：长思维链里，一条很长的回答可能包含大段重复或胡言乱语。按回答平均时，这些 token 的权重被长度稀释，得不到足够的惩罚；按 token 平均后，长回答里每个 token 和短回答的同样重要。见「DAPO 损失」一题。
- **`token-mean` 和梯度累积、数据并行**：`token-mean` 的分母 $N$ 是整个 batch 的 token 数。如果每个 micro-batch 各自除以自己的 token 数、再对 micro-batch 取平均，结果就不再是全局的 token-mean，token 少的 micro-batch 被放大。这和「SFT 损失：助手掩码与样本打包」一题里 2024 年梯度累积的 bug 是同一个问题。TRL 的 `"dapo"` 会先统计全局 batch 的有效 token 数再归一化，`"bnpo"` 则只在本地 batch 内归一化。
- **常数因子不改变方向**：`seq-mean-token-sum` 和 `seq-mean-token-sum-norm` 只差一个常数 $L$，梯度方向相同，只是等效学习率不同。不同框架选的常数也不一样，比如 TRL 的 `"dr_grpo"` 除以 $B$ 乘最大生成长度，换框架时要注意学习率是否需要调整。而前三种模式之间是**重新分配权重**，会改变优化的目标本身。

## 面试追问

1. **GRPO 原始的聚合方式为什么会带来长度偏差？**
   它先在每个回答内按 token 平均，等于给每个 token 乘了 $1/n_i$。正确回答越短，每个 token 被推得越多；错误回答越长，每个 token 被罚得越少，模型因此倾向于把错误的回答写长。
2. **`token-mean` 和 `seq-mean-token-mean` 各自适合什么场景？**
   回答长度差不多时两者接近。长思维链 RL 里长度差别很大，`token-mean` 让每个 token 权重相同，长回答里的坏模式能被有效惩罚，DAPO 和现在框架的默认值都用它。`seq-mean-token-mean` 让每个回答权重相同，更接近「一条回答一个样本」的直觉。
3. **Dr. GRPO 为什么要除以一个常数，而不是用 `token-mean`？**
   `token-mean` 的权重 $1/N$ 仍然依赖这个 batch 恰好有多少 token，batch 之间会波动。除以常数后，每个 token 的权重完全不受长度和 batch 组成影响，梯度就是无偏的策略梯度乘一个固定系数。
4. **数据并行、多个 micro-batch 时，`token-mean` 要注意什么？**
   分母应该是全局的有效 token 数。每个 micro-batch 各自平均再取平均，会放大 token 少的 micro-batch。正确做法是先求和，最后统一除以全局 token 数（或先 all-reduce token 数）。
5. **换一种聚合方式，需要重新调学习率吗？**
   通常需要。不同模式的损失量级不同：`seq-mean-token-sum` 会随回答长度放大，除以常数的版本取决于选了多大的 $L$。只差常数因子的模式之间，相当于学习率乘了一个系数。
