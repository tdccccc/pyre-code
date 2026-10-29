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

**通信量**：all-reduce 本身就可以分解为 reduce-scatter + all-gather，所以 FSDP 的通信量和普通数据并行相当，换来的是显存大幅下降。

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
