## 背景

**数据并行**（DDP）里每张卡都保存一份完整的参数、梯度和优化器状态，卡越多浪费越多。用 Adam 时，每个参数在 fp32 下大约要占

$$
\underbrace{4}_{\text{参数}} + \underbrace{4}_{\text{梯度}} + \underbrace{8}_{\text{Adam 的 } m, v} = 16 \text{ 字节}
$$

一个 7B 模型就要 112 GB，单卡放不下。

**FSDP**（Fully Sharded Data Parallel，对应 ZeRO-3）把这三样都**切成 $N$ 份**，每张卡只长期保存自己那一份，显存约降为 $1/N$。需要完整参数时临时收集，用完就丢。

## 核心公式

参数拼接为 $\theta = [\theta_1 ; \theta_2 ; \dots ; \theta_N]$，卡 $i$ 只保存 $\theta_i$。

**All-gather**：每张卡贡献自己的分片，每张卡最后都得到完整的拼接结果：

$$
\text{卡 } i \text{ 持有 } \theta_i \quad\xrightarrow{\text{all-gather}}\quad \text{每张卡都有 } \theta = [\theta_1 ; \dots ; \theta_N]
$$

**Reduce-scatter**：每张卡 $r$ 用自己的数据算出一份完整梯度 $g^{(r)}$；先对所有卡求和（reduce），再切开分发（scatter），卡 $i$ 只拿到第 $i$ 块：

$$
\text{卡 } i \text{ 得到 } \sum_{r=1}^{N} g^{(r)}_i
$$

**更新**：每张卡只更新自己的分片（本题用 SGD，$\eta = 0.01$）：

$$
\theta_i \leftarrow \theta_i - \eta\, g_i
$$

把所有分片拼起来，结果和不分片时对完整参数做 $\theta \leftarrow \theta - \eta\, g$ 完全一样。

**通信量**：all-reduce 本身就可以分解为 reduce-scatter + all-gather，本题模拟的一次 all-gather 加一次 reduce-scatter 与普通数据并行相当。真实的 FSDP 为了用完即弃，反向时还要再 all-gather 一次参数，总量约为数据并行的 $1.5$ 倍（见延伸）；换来的是显存大幅下降。

本题只模拟一份数据：梯度由 `grad_fn` 对完整参数一次算出，reduce-scatter 退化为把梯度切成 $N$ 块。

## 形状

| 量 | 形状 |
| --- | --- |
| 每个 `param_shards[i]` | 一维，长度 $n_i$ |
| all-gather 后的完整参数 | 一维，长度 $\sum_i n_i$ |
| 梯度 | 与完整参数相同 |
| 第 $i$ 块梯度 | 长度 $n_i$，必须与第 $i$ 个分片对齐 |

## 用到的 PyTorch 操作

- `torch.cat(shards)`：拼接（模拟 all-gather）。
- `t.chunk(n)`：切成 $n$ 块；`t.split(sizes)`：按指定长度切。
- 列表推导式生成新的分片列表。

## 常见坑

- **梯度块和参数分片对不齐**：`chunk(n)` 按 $\lceil L/n \rceil$ 切块，分片长度不均匀时可能和原来的分片长度不一致。更稳妥的是按每个分片的实际长度 `split([s.numel() for s in shards])`。
- **原地修改了传入的分片**：题目要求返回更新后的分片列表。
- **返回了完整参数**：要返回 $N$ 个分片组成的列表。
- 学习率是 $0.01$。

## 延伸

- **真实的 FSDP 不是整个模型一次收集**：模型被切成若干个 FSDP 单元（通常一个 Transformer 层一个）。一个单元的流程是：

  1. 前向之前 all-gather 这一单元的参数，算完立即释放，只留下自己的分片；
  2. 反向之前再 all-gather 一次；
  3. 这一单元的梯度算完后 reduce-scatter，每卡只留自己那一段；
  4. 优化器只更新本地分片，优化器状态也只存本地分片。

  任何时刻只有少数几个单元的完整参数在显存里，所以峰值显存约为「分片后的模型状态 + 最大单元的完整参数 + 激活」。
- **通信量是数据并行的 1.5 倍**：每个参数每步被通信三次，前向 all-gather、反向 all-gather、梯度 reduce-scatter，各约 $\Psi$ 个元素，共约 $3\Psi$；普通 DDP 的 all-reduce 是 $2\Psi$。如果前向后不释放参数（对应 ZeRO-2），可以省掉第二次 all-gather，但就要一直存着完整参数。
- **靠重叠藏住通信**：当前单元计算时，提前发起下一单元的 all-gather（预取）；反向时 reduce-scatter 与后面层的计算同时进行。每卡 token 越多，计算越长，越容易盖住。
- **分片策略**：PyTorch 的 `FULL_SHARD` 对应 ZeRO-3，`SHARD_GRAD_OP` 对应 ZeRO-2，`HYBRID_SHARD` 在机内分片、机间复制，把 all-gather 和 reduce-scatter 限制在 NVLink 内，只剩机间的梯度 all-reduce。
- **混合精度**：分片里存 fp32 主权重，all-gather 前先转成 bf16，通信量因此减半；梯度也可以用低精度做 reduce-scatter。
- **梯度累积的代价**：每个 micro-batch 都要重新 all-gather 一遍参数，累积越多通信越多。可以推迟梯度的 reduce-scatter，只在最后一个 micro-batch 同步，代价是累积期间要存未分片的梯度。
- **和本题的差别**：本题只有一份梯度，reduce-scatter 退化成切块；真实情形下各卡用不同数据算出不同的梯度，reduce-scatter 先对它们求和（框架再除以卡数取平均）再切开。

## 面试追问

1. **FSDP 和 DDP 的区别是什么？**
   DDP 每卡保存完整的参数、梯度、优化器状态，只同步梯度；FSDP 把三者都切开，需要时才临时收集完整参数。显存降到约 $1/N$，代价是通信变成约 $1.5$ 倍，并且要靠预取和重叠隐藏。
2. **FSDP 的通信量是多少？为什么比 DDP 多？**
   每步约 $3\Psi$：前向 all-gather 参数、反向 all-gather 参数、梯度 reduce-scatter；DDP 是 $2\Psi$。多出来的一次 all-gather 是因为前向用完就把参数丢掉了，反向还得再收一次。
3. **FSDP 的峰值显存由什么决定？**
   由分片后的状态（约 $16\Psi/N$）、同时在显存里的完整参数（最大的单元，加上预取的下一单元）、激活三部分决定。单元切得越细，峰值参数越小，但通信次数越多、每次越小，效率会下降。
4. **FSDP 和 ZeRO-3 是什么关系？**
   思路相同：把参数、梯度、优化器状态全部切开。ZeRO 是 DeepSpeed 里的名字，FSDP 是 PyTorch 原生的实现。`SHARD_GRAD_OP` 相当于 ZeRO-2。
5. **为什么 FSDP 反而在每卡 batch 很小时效率差？**
   通信量只与参数量有关，与 token 数无关；计算量与 token 数成正比。每卡 token 太少时计算时间太短，盖不住 all-gather，GPU 就在等通信。
