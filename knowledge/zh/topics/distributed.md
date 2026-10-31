## 解决什么问题

训练大模型会撞上三堵墙：

- **模型状态放不下**：混合精度 + Adam 每个参数约 $16$ 字节，7B 模型就是 $112$ GB（账本见「性能估算」专题）。
- **激活放不下**：序列越长，激活越大。
- **算得太慢**：一张卡的算力撑不起所需的总 FLOPs。

分布式训练的所有方法都在回答同一个问题：**把哪个维度切到多张卡上，切完之后用什么通信把结果拼回来**。可以切的维度有：数据（batch）、模型状态、层内的权重矩阵、层、序列、专家。

判断一种切法好不好，用一个很粗但很管用的代价模型。设 $\alpha$ 是每条消息的固定延迟，$B$ 是链路带宽：

$$
T_{\text{step}} \approx T_{\text{计算}} + T_{\text{通信}}^{\text{未被重叠的部分}},
\qquad
T_{\text{通信}} \approx n_{\text{消息}}\,\alpha + \frac{\text{字节数}}{B}
$$

所以每种并行方式要看三件事：**通信字节数**、**能不能和计算重叠**、**走的是 NVLink 还是机间网络**。

## 通信原语

设 $W$ 张卡，$L$ 是完整张量的元素数。下面的发送量按环形（ring）实现计：

| 原语 | 起点 → 终点 | 每卡发送量 | 主要出处 |
| --- | --- | --- | --- |
| all-gather | 每卡 $L/W$ → 每卡完整 $L$ | $\frac{W-1}{W}L$ | FSDP 取参数 |
| reduce-scatter | 每卡完整 $L$ → 每卡 $L/W$ 的和 | $\frac{W-1}{W}L$ | ZeRO 归约梯度 |
| all-reduce | 每卡完整 $L$ → 每卡完整的和 | $\frac{2(W-1)}{W}L$ | DP 梯度、TP 激活 |
| all-to-all | 每卡发 $L/W$ 给每个其他卡 | $\frac{W-1}{W}L$ | Ulysses、EP |
| send / recv | 点对点 | $L$ | PP 激活、Ring 的 K/V |

**all-reduce = reduce-scatter + all-gather**。reduce-scatter 做完，每卡拿着一段求好和的结果；all-gather 再把各段发给所有人。两段各发 $\frac{W-1}{W}L$，合起来就是 $\frac{2(W-1)}{W}L$，当 $W$ 很大时趋近 $2L$，与卡数无关。直观的下界理由：每张卡自己那份数据里，有 $\frac{W-1}{W}$ 属于别的卡负责的段，必须发出去；最终结果里也有 $\frac{W-1}{W}$ 是别的卡算的，必须收进来。ring 恰好达到这个量，所以说它**带宽最优**。

代价是延迟：一次 ring all-reduce 有 $2(W-1)$ 步，$\alpha$ 项随 $W$ 线性增长，小消息或卡数很多时要换树形等对数步数的算法（NCCL 会按消息大小和拓扑自动选择）。

**all-to-all** 和 reduce-scatter 的每卡发送量相同，但每对卡之间都有数据要交换，吃的是整个网络的对分带宽，不像 ring 只用相邻链路。所以它在跨机时往往比 ring 类集合通信更难受。

## 对比

约定：$\Psi$ 是参数量，$n$ 是一张卡上的 token 数（$n = b \cdot s$），$h$ 是隐藏维，$T$ 是一条完整序列的 token 数（含 batch），$k$ 是 MoE 的 top-k。

**切什么、用什么通信**。DP 与 ZeRO 两行是每个训练步的量（$W$ 大时的极限）；其余各行是每层、每个 micro-batch、前向一遍的量，反向再来一份同量级的。

| 方式 | 切什么 | 通信 | 每卡通信量（元素） | 适合放哪 |
| --- | --- | --- | --- | --- |
| DP | 数据 | all-reduce 梯度 | $\approx 2\Psi$ | 最外层，可跨机 |
| ZeRO-1 / 2 | 优化器态（和梯度） | reduce-scatter 梯度 + all-gather 参数 | $\approx 2\Psi$，同 DP | 跨机 |
| ZeRO-3 / FSDP | 参数也切 | 前后向各 all-gather 一次 + reduce-scatter | $\approx 3\Psi$ | 跨机，靠预取重叠 |
| TP | 层内权重矩阵 | all-reduce 激活，前向 2 次 | $\approx 4n h$ | 机内 NVLink |
| PP | 层 | 相邻 stage 点对点 | 每个边界 $n h$ | 可跨机 |
| Ring（CP） | 序列，K/V 绕环 | $P-1$ 轮点对点 | $\approx 2Th$ | 机内；跨机要块够大 |
| Ulysses | 序列 ↔ 头 | 2 次 all-to-all | $\approx 4Th/P$ | 机内 |
| EP | 专家 | 2 次 all-to-all（分发、合并） | $\approx 2k\,n h$ | 机内为佳 |

**对显存的影响**，「全」表示每卡仍保存完整的一份：

| 方式 | 参数 | 梯度 | 优化器态 | 激活 |
| --- | --- | --- | --- | --- |
| DP | 全 | 全 | 全 | 不变 |
| ZeRO-1 | 全 | 全 | $1/W$ | 不变 |
| ZeRO-2 | 全 | $1/W$ | $1/W$ | 不变 |
| ZeRO-3 | $1/W$，另有临时的一层完整参数 | $1/W$ | $1/W$ | 不变 |
| TP | $1/t$ | $1/t$ | $1/t$ | 部分切开，配合 SP 后基本切开 |
| PP | $1/p$ | $1/p$ | $1/p$ | 1F1B 下峰值与 $m$ 无关 |
| Ring / Ulysses | 全 | 全 | 全 | 沿序列 $1/P$ |
| EP | 专家部分 $1/\text{EP}$ | 同左 | 同左 | 取决于路由 |

几点读法：

- **ZeRO 是「切冗余」，不是「切计算」**。每张卡仍然拿完整的层做前向反向，只是不再长期保存别人负责的那份状态，所以激活不变。
- **DP 的通信量与 token 数无关，计算量与 token 数成正比**。放大每卡 token 数（比如梯度累积）就能把它摊薄，也更容易藏在反向计算后面。TP 的通信量则与 token 数成正比，摊不薄。
- **ZeRO-1/2 有显存下限**：$W \to \infty$ 时，ZeRO-1 趋近 $4\Psi$，ZeRO-2 趋近 $2\Psi$；只有 ZeRO-3 能随 $W$ 一直降下去。
- **PP 不省激活，省的是参数和优化器态**。1F1B 下第 0 个 stage 同时有 $p$ 个 micro-batch 在途，每个只经过自己那 $L/p$ 层，合起来相当于一个 micro-batch 过完整个模型的激活。它保证的是激活不随 $m$ 增长，而不是比不并行更省。
- **TP 切不掉全部激活**。按 Megatron 序列并行论文的口径，每层激活约为 $sbh\,(10 + 24/t + 5as/(ht))$，其中 $10\,sbh$ 一项（LayerNorm、dropout、残差等）在 TP 组内每卡一份；加上序列并行（SP，沿序列维切这些区域）后整体变成 $1/t$。

## 7B 量级的几笔账

取 $\Psi = 7 \times 10^9$，模型状态按 $16\Psi$ 计（不含激活），单位 GB：

| 方式 | $W=8$ | $W=64$ |
| --- | --- | --- |
| DP | 112 | 112 |
| ZeRO-1 | 38.5 | 29.3 |
| ZeRO-2 | 26.3 | 15.5 |
| ZeRO-3 | 14.0 | 1.75 |

**DP 的梯度 all-reduce**：bf16 梯度 $14$ GB，每卡约发送 $2 \times 14 = 28$ GB。按 H100 的 NVLink 单向约 $450$ GB/s 算，约 $62$ ms；若只有一块 $400$ Gb/s 的网卡（约 $50$ GB/s），约 $0.56$ s。如果每卡每步 $16384$ 个 token，计算量 $6\Psi n \approx 6.9 \times 10^{14}$ FLOPs，按 $40\%$ MFU 约 $1.7$ s，通信可以藏在反向后面。

**TP 的激活 all-reduce**：LLaMA-7B 规模（$h = 4096$），$n = 4096$，$t = 8$。一次 all-reduce 的张量是 $4096 \times 4096 \times 2$ B $\approx 34$ MB，每卡发送约 $59$ MB；一层前后向共 $4$ 次，约 $235$ MB，NVLink 上约 $0.5$ ms。同一层前后向的计算（只算权重矩阵乘，$6 \times 12h^2 \times n / t$ FLOPs，$40\%$ MFU）约 $1.6$ ms。也就是说 NVLink 上通信已占计算的三分之一左右，而且它在关键路径上、难以重叠；换成上面的网卡，约 $4.7$ ms，通信比计算还长。这就是 TP 基本只在机内用的原因。

这些数字是估算，用来判断量级，不是实测。

## 流水线：GPipe、1F1B 与 interleaved

设 $p$ 个 stage、$m$ 个 micro-batch，每个 stage 前向 $t_f$、反向 $t_b$。流水线要多出 $p-1$ 份「填充 + 排空」的时间，总时间和气泡率为

$$
T = (m + p - 1)(t_f + t_b),
\qquad
\text{bubble} = \frac{p-1}{m+p-1}
$$

这个式子不要求 $t_f = t_b$，但假设各 stage 耗时均衡。若以有用计算时间 $m(t_f+t_b)$ 为分母，则气泡是 $\frac{p-1}{m}$。读论文时注意分母是哪个。

**interleaved 1F1B**：每张卡不再放连续的一段层，而是放 $v$ 个小块（模型被切成 $pv$ 块，轮流分配），每块时间缩为 $1/v$，填充和排空的时间随之缩为 $1/v$：

$$
T = \Bigl(m + \frac{p-1}{v}\Bigr)(t_f + t_b),
\qquad
\text{bubble} = \frac{p-1}{vm + p - 1}
$$

代价是 stage 之间的点对点通信量变成原来的 $v$ 倍。

以 $p = 8$、$m = 32$ 为例：

| 方式 | 气泡率 | 激活与额外代价 |
| --- | --- | --- |
| GPipe | $17.9\%$ | 每个 stage 同时在途 $m = 32$ 个 micro-batch |
| 1F1B | $17.9\%$ | stage $s$ 在途最多 $\min(8-s,\ 32)$ 个，最多 8 |
| interleaved，$v=2$ | $9.9\%$ | 点对点通信变为 $2$ 倍 |

**1F1B 省的是激活显存，不是气泡**：它只是调整了每个 stage 内部前向反向的先后顺序，总时间不变。

## Ring Attention 与 Ulysses

两者都是为长序列把激活沿序列切到 $P$ 张卡上，但让每个 token 看到整条序列的办法相反：Ring 让 **K/V 块动**（Q 不动），Ulysses 让**切分方式变**（序列切分换成按头切分）。

| | Ring | Ulysses |
| --- | --- | --- |
| 通信 | $P-1$ 轮点对点传 K/V | 2 次 all-to-all |
| 每卡通信量（前向，MHA） | $\approx 2Th$，不随 $P$ 减小 | $\approx 4Th/P$，随 $P$ 减小 |
| 并行度上限 | 不受头数限制 | $P \le H$（GQA 时为 KV 头数） |
| 与计算重叠 | 天然可以，要求块够大 | 较难，all-to-all 在关键路径上 |
| 注意力计算 | 需要在线 softmax 合并 | 任何注意力 kernel 原样可用 |
| 因果掩码 | 负载不均 | 无此问题 |

**通信量怎么来**。Ring 每轮把 K 和 V 各一块（$T/P$ 个 token）传给下一张卡，共 $P-1$ 轮，每卡共发 $2\frac{P-1}{P}Th$，GQA 下 $h$ 换成 K/V 的总宽度，更小。Ulysses 第一次 all-to-all 处理 Q、K、V 三个张量，第二次处理输出一个张量，共 $4$ 份，每份每卡持有 $Th/P$，发出其中 $\frac{P-1}{P}$。

取 $T = 131072$、$h = 4096$、bf16，一层前向每卡发送：

| $P$ | Ring | Ulysses |
| --- | --- | --- |
| 8 | 约 1.9 GB | 约 0.47 GB |
| 16 | 约 2.0 GB | 约 0.25 GB |

**Ring 能否被计算盖住**。每轮 attention 的计算量约为 $4\,c^2 h$ FLOPs（$c = T/P$ 是每卡 token 数，$QK^\top$ 和 $AV$ 各 $2c^2h$），传输量为 $2ch$ 个元素（bf16 下 $4ch$ 字节）。设峰值算力 $F$、单向带宽 $B$，计算时间不短于传输时间的条件是

$$
\frac{4\,c^2 h}{F} \ \ge\ \frac{4\,c h}{B}
\quad\Longleftrightarrow\quad
c \ \ge\ \frac{F}{B}
$$

按 H100 的 $F \approx 989$ TFLOP/s 估算：NVLink 单向 $450$ GB/s 时 $c \gtrsim 2 \times 10^3$ 个 token；单块 $50$ GB/s 的网卡时 $c \gtrsim 2 \times 10^4$。这是按峰值算力算的；真实 kernel 达不到峰值，条件会宽松一些；而因果掩码让计算量约减半，条件又收紧。

**因果掩码下 Ring 的负载不均**：连续切分时，只有最后一张卡要看到所有块，第一张卡只需要自己的那一块。常见补救是打乱切分方式（交错或「之字形」排布序列），让每张卡拿到的因果工作量接近。Ulysses 每张卡拿到的是完整序列的一部分头，掩码照常使用，没有这个问题。

**怎么组合**：Ulysses 的并行度受头数限制，而 TP 也在切头，两者共用同一批头；Ring 不受此限。常见做法是节点内用 Ulysses，节点间用 Ring，叠成二维。

## 演进

1. **数据并行与 ring all-reduce**：最早的做法是让一个参数服务器汇总梯度，它的链路是瓶颈。2017–2018 年前后 ring all-reduce 经由 NCCL、Horovod 等进入主流框架，DDP 还把梯度分桶，边反向边通信，把通信盖在计算后面。
2. **按层切与按矩阵切**：GPipe（2018）用 micro-batch 填充流水线；PipeDream 提出 1F1B 的调度思路；Megatron-LM（2019）给出 Transformer 的张量并行切法。
3. **ZeRO 与 FSDP**：ZeRO（2019）发现数据并行里的状态是冗余的，按阶段把优化器态、梯度、参数依次切开；PyTorch 把同样的思路做成原生的 FSDP（2022 年进入 PyTorch 1.11）。
4. **3D 并行与序列并行**：Megatron 把 TP、PP、DP 组合起来（2021），并提出 interleaved 1F1B；序列并行（2022）把 TP 没切开的 LayerNorm、dropout 激活沿序列切开，通信量不变。
5. **MoE 与专家并行**：GShard（2020）、Switch Transformer（2021）用固定容量的专家缓冲区，让 all-to-all 和专家计算的形状固定。
6. **长上下文**：Ring Attention、DeepSpeed-Ulysses（均为 2023）把序列并行用到注意力上，现在常叠加使用。
7. **进一步压气泡**：zero bubble 一类调度把反向拆成「对输入的梯度」和「对权重的梯度」两部分，用后者去填气泡。

## 3D / 4D 并行怎么组合

总卡数是各维度度数之积：

$$
N_{\text{GPU}} = \text{DP} \times \text{PP} \times \text{TP}\ (\times\ \text{CP}\ \times\ \text{EP}\cdots)
$$

**摆放原则：通信越频繁、越难重叠的维度，越要放在越快的链路上。**

| 维度 | 通信特点 | 放置 |
| --- | --- | --- |
| TP | 每层 4 次 all-reduce，在关键路径上 | 机内，度数不超过一台机器的卡数（常见 8） |
| CP / Ulysses / EP | 每层的 all-to-all 或 K/V 环 | 尽量机内 |
| PP | 只在 stage 边界传一次激活，量小 | 机间 |
| DP | 每步一次，可与反向重叠 | 最外层，机间 |

示意：$1024 = 8\ (\text{TP}) \times 16\ (\text{PP}) \times 8\ (\text{DP})$。70B 模型加上 DP 维的 ZeRO-1，每卡模型状态约

$$
\frac{(4 + 12/\text{DP})\,\Psi}{\text{TP}\times\text{PP}} = \frac{5.5 \times 70 \times 10^9}{128} \approx 3.0 \text{ GB}
$$

剩下的显存主要留给激活。

**几个实际的约束**：

- **全局 batch 把 DP 和气泡绑在一起**。全局 batch $=$ DP $\times\, m \times$ micro-batch 大小。全局 batch 固定为 $1024$ 条序列、micro-batch 为 $1$、$p = 16$：DP $= 8$ 时 $m = 128$，气泡 $15/143 \approx 10.5\%$；DP $= 64$ 时 $m = 16$，气泡 $15/31 \approx 48\%$。
- **PP 通常只搭配 ZeRO-1**。ZeRO-2/3 要在每个 micro-batch 上做梯度或参数的分片通信，而 PP 的 micro-batch 很多，通信被成倍放大。
- **EP 常复用 DP 这一维**：非专家层在这些卡上是数据并行，专家层是专家并行。
- **名称容易混**：「SP」有两个含义。Megatron 的序列并行附着在 TP 组内，不占额外的卡；长序列的序列并行（Ring、Ulysses，Megatron 里称 context parallel）是另一个独立的维度。

## 怎么选

从简单的开始，装不下或太慢再加一维：

1. **模型状态单卡放得下**：纯 DP；状态吃紧时加 ZeRO-1/2，通信量与 DP 相同，白赚显存。
2. **状态放不下，但卡数不多、互联快**：FSDP / ZeRO-3。代码改动最小，不需要改模型结构。通信是 DP 的 $1.5$ 倍，要靠预取与计算重叠，每卡 token 数不能太小。
3. **模型再大，或跨很多机器**：机内 TP，机间 PP，外层 DP（加 ZeRO-1）。需要调 micro-batch 数 $m$ 去压气泡。
4. **序列很长**：加 Ulysses 或 Ring。头数够、都在机内，优先 Ulysses；度数超过头数或必须跨机，用 Ring。
5. **MoE 模型**：加 EP；路由负载不均时用容量系数限制最坏情况，或者改用不丢 token 的实现。
6. **网络慢**：减少对高带宽的依赖，避免跨机 TP；FSDP 可只在机内分片、机间复制（PyTorch 的 hybrid sharding）。

## 题目顺序

1. **环形 All-Reduce**：先把通信原语弄清楚，reduce-scatter 与 all-gather 是后面所有题的基本部件。
2. **ZeRO 优化器**：用 reduce-scatter + all-gather 实现分片的 Adam 步，理解为什么优化器状态可以切开。
3. **FSDP**：在 ZeRO 的基础上把参数也切开，看参数怎样临时收集、用完即弃。
4. **张量并行**：从切数据和状态转到切权重矩阵，列切加行切，只需一次 all-reduce。
5. **1F1B 流水线**：按层切，重点是调度顺序、气泡率和激活显存。
6. **Ring Attention**：转到序列维，用在线 softmax 在环上合并注意力。
7. **Ulysses**：同样是序列并行，换成 all-to-all，与上一题对比通信模式和对头数的约束。
8. **专家分发**：最后是 EP 的核心，固定容量的缓冲区与 all-to-all 的 token 路由。

## 面试追问

1. **ZeRO-1、2、3 分别切什么？通信量有什么区别？**
   ZeRO-1 切优化器态，ZeRO-2 再切梯度，ZeRO-3 再切参数；每卡模型状态依次是 $4\Psi + 12\Psi/W$、$2\Psi + 14\Psi/W$、$16\Psi/W$（混合精度 + Adam）。ZeRO-1/2 用 reduce-scatter 加 all-gather，总量等于一次 all-reduce，与 DP 相同；ZeRO-3 前向和反向各要 all-gather 一次参数，再加一次 reduce-scatter，共约 $3\Psi$，是 DP 的 $1.5$ 倍。

2. **为什么说 all-reduce 等于 reduce-scatter 加 all-gather？ring 的带宽项为什么是 $2(W-1)/W$？**
   两段各把数据环绕一圈的 $(W-1)/W$，所以合起来是 $2(W-1)/W \cdot L$，卡数再多也不超过 $2L$。每张卡必须发出不归自己负责的 $(W-1)/W$，也必须收进别人算好的 $(W-1)/W$，所以这是下界，ring 达到了它。代价是延迟项随 $W$ 线性增长。

3. **TP 一层要几次通信？为什么只在机内用？**
   前向 $2$ 次 all-reduce（注意力输出投影之后、MLP 第二层之后），反向再 $2$ 次，共 $4$ 次。通信量与 token 数成正比，且在计算的关键路径上，难以重叠，所以必须放在 NVLink 这类高带宽链路上。

4. **1F1B 比 GPipe 好在哪里？它能降低气泡吗？**
   不能。两者气泡率都是 $(p-1)/(m+p-1)$。1F1B 的好处是激活显存：每个 stage 同时在途的 micro-batch 数最多 $p$（stage $s$ 为 $\min(p-s, m)$），与 $m$ 无关；GPipe 要同时存下 $m$ 个。想压气泡只能增大 $m$、用 interleaved 或 zero bubble 一类调度。

5. **Ring Attention 和 Ulysses 怎么选？**
   Ulysses 实现简单、注意力 kernel 原样可用、通信量随 $P$ 减小，但并行度不能超过头数，all-to-all 难以和计算重叠。Ring 不受头数限制，通信可以被计算盖住，但要求块足够大，因果掩码下要处理负载不均。头数够、在机内用 Ulysses；超出头数或要跨机用 Ring，两者也可以叠加。

6. **给你 1024 张 H100（8 卡一台）训一个 70B 模型，并行方案怎么设计？**
   先估模型状态：$16\Psi \approx 1.1$ TB，单卡放不下。TP 取 $8$（一台机器内），PP 取 $16$ 左右跨机，剩下 $8$ 路做 DP 并加 ZeRO-1；每卡模型状态约 $3$ GB。然后核对全局 batch 能否让 $m$ 足够大，把气泡压到可接受的范围，再看激活是否放得下，必要时加重计算或序列并行。序列很长再加 CP 一维。
