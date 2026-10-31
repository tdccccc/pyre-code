## 解决什么问题

深层网络里，每一层的输出尺度会随着训练漂移，层数一多就可能逐层放大或缩小，梯度跟着不稳定，学习率也不敢调大。**归一化**在层与层之间把激活值拉回统一的尺度，再交给可学习的参数恢复表达能力。

所有归一化方法都是同一个公式：

$$
y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

区别只有一个：**$\mu$、$\sigma^2$ 沿哪些维度统计**。RMSNorm 再进一步，去掉了 $\mu$ 和 $\beta$。

## 对比

以 CNN 的输入 `(N, C, H, W)` 为例（Transformer 里对应 `(B, S, D)`，D 相当于 C）：

| 方法 | 每组统计量覆盖 | 依赖 batch | 训练 / 推理一致 | 典型使用 |
| --- | --- | --- | --- | --- |
| BatchNorm | 一个通道，跨 N、H、W | 是 | 否，推理用运行统计量 | ResNet 等 CNN |
| LayerNorm | 一个样本，跨 C（CNN 里常跨 C、H、W） | 否 | 是 | 原始 Transformer、BERT、GPT-2 |
| RMSNorm | 同 LayerNorm | 否 | 是 | LLaMA、Qwen、DeepSeek、Mistral |
| GroupNorm | 一个样本的一组通道，跨 H、W | 否 | 是 | 小 batch 的检测分割、扩散模型的 UNet |
| InstanceNorm | 一个样本的一个通道，跨 H、W | 否 | 是 | 风格迁移 |

后两种本站没有题，面试问到时知道统计范围即可：GroupNorm 的组数为 $1$ 时退化成 LayerNorm，组数等于通道数时退化成 InstanceNorm。

## 演进

1. **BatchNorm（2015）**：CNN 时代的标准组件，让深层 CNN 可以用大学习率稳定训练。
2. **LayerNorm（2016）**：为 RNN 设计，序列长短不一，batch 统计量不可靠，于是改成每个样本自己统计。Transformer 沿用了它。
3. **Pre-LN 取代 Post-LN**：原始 Transformer 把 LN 放在残差相加**之后**，GPT-2 起改到子层**之前**，深层模型训练稳定得多（见下一节）。
4. **RMSNorm（2019）**：发现减均值贡献不大，去掉后更省；LLaMA 之后成为大模型默认选择。
5. **更多位置的归一化**：例如 Qwen3、Gemma 3 对注意力里的 $q$、$k$ 再做一次 RMSNorm（QK-Norm），防止注意力分数过大；DiT 用条件向量生成 $\gamma$、$\beta$（AdaLN，见「扩散与流匹配」专题）。

## Pre-LN 与 Post-LN

$F$ 表示一个子层（注意力或 MLP），$x_l$ 是第 $l$ 层的输入：

$$
\text{Post-LN: } x_{l+1} = \operatorname{LN}\bigl(x_l + F(x_l)\bigr)
$$

$$
\text{Pre-LN: } x_{l+1} = x_l + F\bigl(\operatorname{LN}(x_l)\bigr)
$$

| | Post-LN | Pre-LN |
| --- | --- | --- |
| 残差通路 | 每层都要经过一次 LN | 从输入到输出是一条不经过 LN 的直通路，梯度可以直达浅层 |
| 训练稳定性 | 层数深时梯度不稳，依赖学习率 warmup 和细致的初始化 | 稳定，深层模型也容易训 |
| 效果 | 训得起来时最终效果往往略好 | 越往深层，残差流的尺度越大，每层的相对贡献变小 |
| 额外要求 | 无 | 最后一层之后要再加一个 LN，再接输出头 |

现在的大模型基本都用 Pre-LN（或 Pre-RMSNorm）。

## 怎么选

- **CNN、batch 较大**：BatchNorm。推理时它能折叠进前面的卷积，没有额外开销。
- **CNN、batch 很小**（检测、分割、高分辨率生成）：GroupNorm。
- **Transformer、序列模型**：LayerNorm 或 RMSNorm；新的大模型默认用 Pre-RMSNorm。

**为什么 Transformer 不用 BatchNorm**：

- 序列长短不一，padding 会混进 batch 统计量。
- 大模型每张卡上的 micro-batch 很小，统计量噪声大；跨卡同步统计量又要额外通信。
- 文本任务里，不同 batch 之间的统计量波动很大，运行统计量很难估准，训练和推理的行为容易对不上。
- 自回归推理一次只生成一个 token，没有 batch 维可以统计。

## 题目顺序

1. **LayerNorm**：最基础的形式，没有状态，先把公式和广播写对。
2. **RMSNorm**：在 LayerNorm 上做减法，对比着写。
3. **BatchNorm**：换一个统计方向，再加上训练和推理两种模式、运行统计量的原地更新。

## 面试追问

1. **BatchNorm 和 LayerNorm 有什么区别？**
   统计方向不同：BN 对每个特征跨样本统计，LN 对每个样本跨特征统计。所以 BN 依赖 batch、训练和推理行为不同，LN 两者都没有。
2. **Pre-Norm 和 Post-Norm 各有什么优缺点？**
   Pre-Norm 的残差通路是直通的，深层训练稳定、不太依赖 warmup，代价是深层的相对贡献变小、需要在最后补一个 LN；Post-Norm 训练难，但训得起来时效果可能略好。
3. **归一化为什么有效？**
   BN 论文最初的解释是减少「内部协变量偏移」（internal covariate shift），后来的研究（Santurkar et al., 2018）认为这个解释站不住，更主要的作用是让损失曲面更平滑、梯度更可预测，从而能用更大的学习率。面试时两种说法都提，并说明后者是目前更被认可的观点。
4. **为什么归一化之后还要乘 $\gamma$、加 $\beta$？**
   强行把每层输出固定为均值 $0$、方差 $1$ 会限制表达能力。$\gamma$、$\beta$ 让网络自己学合适的尺度和偏移，必要时甚至可以学回恒等变换。
5. **为什么现代大模型几乎都用 RMSNorm？**
   效果和 LayerNorm 相当，但少一次求均值、少一组参数，在规模很大时能省下可观的时间。详见 RMSNorm 一题的笔记。
