## 背景

**知识蒸馏**（knowledge distillation，Hinton 等人 2015）让小的学生模型去模仿大的教师模型。教师给出的不只是「正确答案是哪个」，而是一个完整的概率分布，里面带着「哪些词也说得通、哪些完全不对」的信息，比单个标签丰富得多。

用在语言模型上有两种做法：

- **序列级蒸馏**：让教师生成回答，学生在这些文本上做 SFT。只需要教师的输出文本，不要求词表相同。DeepSeek-R1 的蒸馏版小模型就是这样得到的：用 R1 生成的数据直接 SFT，没有再做 RL。
- **token 级蒸馏**（本题）：在同一个序列的每个位置上，让学生的下一个 token 分布去拟合教师的分布。需要教师的 logits，并且两者词表一致。

按数据来源还可以分成**离线**（在固定数据或教师生成的文本上算）和**在线**（on-policy：在学生自己采样出来的序列上，让教师逐 token 打分）。GKD（Agarwal 等人）系统地研究了在线蒸馏和 JSD 损失；Thinking Machines 2025 年的博客 *On-Policy Distillation* 在学生自己的采样上逐 token 计算反向 KL。

## 核心公式

每个位置上，学生和教师的分布（温度 $\tau$ 软化后）：

$$
p_S = \operatorname{softmax}(s / \tau), \qquad p_T = \operatorname{softmax}(t / \tau)
$$

$s, t$ 是该位置的 logits（长度为词表大小 $V$）。KL 散度 $\operatorname{KL}(p \Vert q) = \sum_v p_v (\log p_v - \log q_v)$ 不对称，于是有两个方向：

| | 公式 | 期望在谁的分布下取 | 行为 |
| --- | --- | --- | --- |
| 前向 KL | $\operatorname{KL}(p_T \Vert p_S)$ | 教师 | 教师有概率的地方学生都必须有，倾向**覆盖所有峰**（mode-covering） |
| 反向 KL | $\operatorname{KL}(p_S \Vert p_T)$ | 学生 | 学生只要别把概率放到教师认为不可能的地方，倾向**抓住主峰**（mode-seeking） |

**广义 JSD**（GKD 使用）在两者之间插值：

$$
M = \beta\, p_T + (1 - \beta)\, p_S, \qquad
\operatorname{JSD}_\beta = \beta\, \operatorname{KL}(p_T \Vert M) + (1 - \beta)\, \operatorname{KL}(p_S \Vert M)
$$

- $\beta$ 很小时 $M \approx p_S$，$\operatorname{JSD}_\beta \approx \beta \cdot \operatorname{KL}(p_T \Vert p_S)$，表现像前向 KL；$\beta$ 接近 1 时像反向 KL。
- $\beta = 0.5$ 是标准的 JS 散度，对称且不超过 $\log 2$。

**$\tau^2$ 因子**：对学生 logits 求导，前向 KL 的梯度是 $(p_S - p_T) / \tau$，温度升高后 $p_S - p_T$ 本身也变小，整体量级约按 $1/\tau^2$ 缩小。乘上 $\tau^2$ 让不同温度下的梯度量级可比，和硬标签的交叉熵混合时不用重新调权重。本题沿用 Hinton 的约定乘上 $\tau^2$；TRL 的 `generalized_jsd_loss` 不乘这个因子，对照实现时要注意。

最终损失是在所有计入的 token 上取平均：$L = \tau^2 \cdot \sum m D / \sum m$。

## 形状

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `student_logits`、`teacher_logits` | (B, T, V) | 在 V 维上做 `log_softmax` |
| 每个 token 的散度 $D$ | (B, T) | 在 V 维上求和后得到 |
| `mask` | (B, T) | 只对计入的 token 取平均 |
| 输出 | 标量 | |

## 用到的 PyTorch 操作

- `torch.log_softmax(x, dim=-1)`：稳定地得到对数概率；需要概率时再 `.exp()`。
- `torch.logsumexp(torch.stack([a, b]), dim=0)`（或 `torch.logaddexp(a, b)`）：在对数空间里算 $\log(\beta p_T + (1-\beta) p_S)$，等于两个对数概率各加上 $\log \beta$、$\log(1-\beta)$ 后再合并。
- `F.kl_div(input, target, log_target=True)`：逐元素计算 `target * (log target - input)`，即 $\operatorname{KL}(\text{target} \Vert \text{input})$。参数顺序和直觉相反，是最常见的方向错误来源。
- `t.detach()`：教师是常数。
- `mask.to(dtype)`：bool 掩码转成浮点再乘。

## 常见坑

- **方向写反**：前向 KL 的期望在教师分布下取，权重是 $p_T$。用 `F.kl_div` 时第一个参数是学生的对数概率、第二个是教师的。
- **先 softmax 再取 log**：logits 差很大时 softmax 会下溢成 0，`log(0) = -inf`，再乘 0 得到 `nan`。全程用 `log_softmax`。
- **忘了 $\tau^2$**：温度一高，损失和梯度都变得很小，看起来像学不动。
- **教师没有 detach**：教师 logits 如果来自一个 `requires_grad` 的模型，梯度会一路传回教师。实践中教师前向一般放在 `torch.no_grad()` 里。
- **JSD 的权重放错**：$\beta$ 在 $M$ 里乘的是教师分布，和 KL 项前的权重对应；放反了只在 $\beta = 0.5$ 时碰巧相同。
- **按序列平均**：本题是在整个 batch 的 token 上平均，长短不同的序列每个 token 权重一样（各种平均方式的差别见「损失聚合方式」一题）。

## 延伸

- **和 SFT 损失混合**：经典做法是 $\alpha \cdot \text{CE}(y, p_S) + (1 - \alpha) \cdot \text{KD}$，硬标签保证不跑偏，软标签提供更多信息。
- **显存**：教师和学生的 logits 都是 (B, T, V)，词表十几万时非常大。可以像「分块 LM Head 交叉熵」那样按行分块计算；离线蒸馏时常只存教师每个位置的 top-k 对数概率，用这 k 个词近似分布。
- **词表必须对齐**：token 级蒸馏要求两个模型用同一个分词器。即使分词器相同，不同尺寸的模型也可能把 lm_head 填充到不同的词表大小，要先截到共同的真实词表。
- **离线与在线**：离线蒸馏只在教师（或数据）的分布上训练，推理时学生走到自己的分布里就没见过了（暴露偏差）。在线蒸馏在学生自己的采样上让教师逐 token 打分，相当于一个每个 token 都有密集奖励的 RL，比只在序列结尾给奖励的 RL 信号密得多。
- **选哪个方向**：离线蒸馏常用前向 KL；在线蒸馏常用反向 KL 或 JSD，因为学生容量小，硬去覆盖教师的所有峰反而会把概率摊到不合理的地方。没有放之四海皆准的选择，GKD 论文里不同任务的最优 $\beta$ 也不同。
- **端点**：本题要求 $0 < \beta < 1$。有的实现（如 TRL 的 `generalized_jsd_loss`）把 $\beta = 0$、$\beta = 1$ 分别特判成前向、反向 KL。

## 面试追问

1. **前向 KL 和反向 KL 有什么区别？**
   前向 KL 在教师分布下取期望，教师有概率的地方学生必须覆盖，倾向「摊开」；反向 KL 在学生分布下取期望，只惩罚学生放错地方的概率，倾向「集中」到教师的主峰。
2. **为什么要乘 $\tau^2$？**
   软化后对 logits 的梯度量级约按 $1/\tau^2$ 缩小。乘回 $\tau^2$ 让损失量级不随温度变化，和交叉熵混合时权重不用跟着温度重调。
3. **什么是在线蒸馏，好处是什么？**
   在学生自己生成的序列上让教师逐 token 打分并计算散度。它解决离线蒸馏的暴露偏差：学生在训练中就见到并纠正自己的错误。和 RL 比，它每个 token 都有信号，样本效率高得多。
4. **DeepSeek-R1 的蒸馏版模型是怎么得到的？**
   用 R1 生成的推理数据对 Qwen、Llama 等小模型做 SFT，是序列级蒸馏，不需要 R1 的 logits，也不要求词表一致；报告里没有对这些小模型再做 RL。
5. **教师和学生词表不一样怎么办？**
   token 级蒸馏无法直接用，因为两个分布不在同一个空间里。工程上最常见的是改用序列级蒸馏；也有研究在不同词表之间做对齐的方法，但实现复杂得多。
