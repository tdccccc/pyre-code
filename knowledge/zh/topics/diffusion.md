## 解决什么问题

要生成图像（或视频、音频），就要学会从一个简单分布（高斯噪声）出发，变换成复杂的数据分布。扩散模型和流匹配的做法相同：**先人为定义一条从数据到噪声的路径，再训练网络学会走回来**。把数据 $x_{\text{data}}$ 和噪声 $\epsilon \sim \mathcal{N}(0, I)$ 按比例混合：

$$
x_t = a_t\, x_{\text{data}} + b_t\, \epsilon
$$

这个家族里的方法都是上面这一个式子，区别在两处：

- **路径怎么选**：系数 $(a_t, b_t)$ 取什么形状。扩散模型取 $a_t = \sqrt{\bar\alpha_t}$、$b_t = \sqrt{1 - \bar\alpha_t}$（满足 $a_t^2 + b_t^2 = 1$）；rectified flow 取直线，$a_t$、$b_t$ 之和为 $1$。
- **网络预测什么、怎么走回去**：预测噪声 $\epsilon$、干净样本 $x_0$、$v$ 或速度，再用随机（DDPM）或确定性（DDIM、ODE）的方式迭代去噪。

时间方向的约定因论文而异：扩散模型里 $t = 0$ 是数据、$t = T$ 是噪声；本站流匹配一题里 $t = 0$ 是噪声、$t = 1$ 是数据，读论文时先确认方向。

## 对比

| 方法 | 反向过程 | 网络预测 | 步数 | 典型使用 |
| --- | --- | --- | --- | --- |
| DDPM | 随机，每步加新噪声 | 噪声 $\epsilon$ | 多（原论文 1000） | 原始扩散模型 |
| DDIM | 确定性（$\eta = 0$），可跳步 | 噪声 $\epsilon$ | 几十步 | 扩散模型的快速采样 |
| Flow Matching | 确定性 ODE，欧拉等积分 | 速度 $v$ | 几十步 | SD3、Flux |

横向再看三件事：

| 部件 | 作用 | 对应的题 |
| --- | --- | --- |
| 噪声调度 | 决定各时刻的 $\bar\alpha_t$（信噪比） | 噪声调度 |
| 采样器 | 把网络输出变成下一个 $x$ | DDIM、流匹配（采样部分） |
| 条件注入 | 让网络知道时间步和类别、文本 | AdaLN-Zero |

## 前向加噪与信噪比

高斯的线性组合仍是高斯，所以任意时刻的 $x_t$ 可以直接由 $x_0$ 一步算出：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,
\qquad
\operatorname{SNR}(t) = \frac{\bar\alpha_t}{1 - \bar\alpha_t}
$$

$\bar\alpha_t$ 是信号保留的比例，SNR 是信号与噪声的功率之比。训练时随机抽 $t$、抽 $\epsilon$、一步加噪、让网络去预测，不必真的走 $t$ 步。

线性调度下，$\bar\alpha_t$ 降得太快，后段的许多步已经接近纯噪声，几乎白训；余弦调度直接规定 $\bar\alpha_t$ 缓慢下降，让各个噪声水平都有训练样本。

## 三种预测目标的换算

网络可以预测 $\epsilon$、$x_0$ 或 $v$。三者只差一个线性变换，已知 $x_t$ 和 $\bar\alpha_t$ 时可以互相换算。其中 $v$ 的定义（Salimans & Ho，2022，用于渐进式蒸馏）：

$$
v = \sqrt{\bar\alpha_t}\, \epsilon - \sqrt{1 - \bar\alpha_t}\, x_0
$$

| 已知 | 求 $x_0$ | 求 $\epsilon$ |
| --- | --- | --- |
| $\epsilon$ | $\dfrac{x_t - \sqrt{1 - \bar\alpha_t}\,\epsilon}{\sqrt{\bar\alpha_t}}$ | 本身 |
| $x_0$ | 本身 | $\dfrac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1 - \bar\alpha_t}}$ |
| $v$ | $\sqrt{\bar\alpha_t}\,x_t - \sqrt{1 - \bar\alpha_t}\,v$ | $\sqrt{1 - \bar\alpha_t}\,x_t + \sqrt{\bar\alpha_t}\,v$ |

不同参数化的差别在于**对各个噪声水平的损失权重**。以 $x_0$ 的误差为基准：

$$
\lVert \epsilon - \hat\epsilon \rVert^2 = \operatorname{SNR}\cdot \lVert x_0 - \hat x_0 \rVert^2
$$

$$
\lVert v - \hat v \rVert^2 = (\operatorname{SNR} + 1)\cdot \lVert x_0 - \hat x_0 \rVert^2
$$

- $\epsilon$ 预测：低噪声（SNR 高）的时刻权重大，高噪声（SNR 接近 $0$）的时刻权重几乎为 $0$。而 $\bar\alpha_t \to 0$ 时，由 $\epsilon$ 换算 $x_0$ 要除以接近 $0$ 的数，误差被放大，所以 $\epsilon$ 预测在终端信噪比为 $0$ 的调度下不再适用。
- $v$ 预测：权重 $\operatorname{SNR} + 1$ 在两端都不会退化，终端信噪比为 $0$ 时仍然成立。部分 Stable Diffusion 2.x 模型使用了 $v$ 预测。
- 流匹配的速度目标 $x_1 - x_0$ 也是 $x_{\text{data}}$ 和 $\epsilon$ 的线性组合，同属这一族，只是路径不同。

## 采样：从 DDPM 到 DDIM 到 ODE

**DDPM**（2020）一步反向采样（$\alpha_t = 1 - \beta_t$，$z \sim \mathcal{N}(0, I)$）：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\epsilon_\theta\Bigr) + \sigma_t z
$$

每步加入随机项 $\sigma_t z$，且相邻步的间隔必须很小，才能把反向转移近似成高斯，所以要走接近 $T$ 步。

**DDIM**（Song 等，2020 年提出，ICLR 2021）把反向过程写成带参数 $\eta$ 的一族：

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\;\epsilon_\theta + \sigma_t z
$$

$$
\sigma_t = \eta \sqrt{\tfrac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}}\sqrt{1 - \tfrac{\bar\alpha_t}{\bar\alpha_{t-1}}}
$$

$\eta = 1$ 时与 DDPM 的随机性一致，$\eta = 0$ 时 $\sigma_t = 0$，整个过程确定性，也就是 DDIM 一题实现的那一步。它们共享同一个训练目标（边缘分布 $q(x_t \mid x_0)$ 不变），所以 **DDPM 训好的模型可以直接换成 DDIM 采样，不用重训**。确定性之后，相邻两步不必紧挨着，可以只取 $T$ 个时间步里的一个子序列。

**ODE 视角**：$\eta = 0$ 的 DDIM 可以看作对一个常微分方程（概率流 ODE）的一阶离散。流匹配则是直接学这个 ODE 的速度场，用欧拉等方法积分。两者的采样器都是 ODE 求解器，更高阶的求解器（如 DPM-Solver）可以把步数压得更低。

## 条件生成：classifier-free guidance

要按类别或文本生成，网络多输入一个条件 $c$。训练时以一定概率把 $c$ 换成「空条件」$\varnothing$，让同一个网络既会条件预测，也会无条件预测。采样时把两个预测外推：

$$
\tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + w\,\bigl(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\bigr)
$$

$w$ 是引导强度（guidance scale）：$w = 1$ 就是普通的条件预测，$w > 1$ 把结果往条件方向推得更远，样本更贴合条件、更清晰，代价是多样性下降。代价是每一步要多跑一次网络（两次前向可以拼成一个 batch）。同样的外推可以用在速度场上，思路相同。

## 骨干网络：从 UNet 到 DiT

早期扩散模型的去噪网络是 **UNet**：卷积组成的下采样、上采样结构，用 GroupNorm 做归一化（见「归一化」专题），时间步嵌入加到每个残差块里，文本条件通过交叉注意力注入。

**DiT**（Peebles & Xie，2022 年提出，ICCV 2023）把 UNet 换成 Transformer：

1. 先用 VAE 把图像压到潜空间，在潜空间做扩散（latent diffusion，Stable Diffusion 的做法）。
2. 把潜空间特征切成 patch，每个 patch 作为一个 token（和 ViT 相同，见「卷积与视觉」专题）。
3. 一串 Transformer 块处理这些 token；时间步和类别通过 **AdaLN-Zero** 注入每个块。

好处是继承了 Transformer 成熟的扩展规律：加深、加宽、增加 token 数，效果可预测地变好。条件注入上，DiT 论文比较了拼接 token、交叉注意力、AdaLN、AdaLN-Zero 几种方式，AdaLN-Zero 效果最好。SD3 的 MMDiT 则让文本 token 和图像 token 各有一套权重、在注意力里拼在一起，是这条路线的延伸。

## 演进

1. **DDPM（2020）**：证明了去噪训练可以生成高质量图像，代价是上千步的采样。
2. **DDIM（2020）**：同一个模型换成确定性采样，可以跳步，步数降一个数量级。
3. **改进调度（2021）**：Improved DDPM 提出余弦调度，调度开始被当作可设计的对象。
4. **classifier-free guidance（2021–2022）**：不需要额外的分类器，用同一个网络的条件、无条件预测外推，成为文生图的标配。
5. **Latent diffusion（2021–2022）**：在 VAE 的潜空间里做扩散，大幅降低算力，Stable Diffusion 由此而来。
6. **DiT（2022–2023）**：骨干从 UNet 换成 Transformer，用 AdaLN-Zero 注入条件。
7. **流匹配 / rectified flow（2022–）**：不再设计 $\bar\alpha_t$，直接学直线路径上的速度，SD3、Flux 等使用。

## 怎么选

- **只有一个训好的 $\epsilon$ 预测模型，想快**：DDIM 或更高阶的 ODE 求解器，把步数降到几十步。
- **从头训一个大的文生图 / 文生视频模型**：流匹配加 Transformer 骨干是目前的主流选择，路径简单，没有调度和终端信噪比的包袱。
- **继续用扩散参数化**：使用 $v$ 预测，配合终端信噪比为 $0$ 的调度。
- **要更强地贴合条件**：加 classifier-free guidance，调 $w$ 在贴合度和多样性之间折中。

## 题目顺序

1. **噪声调度**：整个家族的基础，先弄清 $\bar\alpha_t$ 和信噪比，后面所有公式都用它。
2. **DDIM 采样步骤**：用 $\bar\alpha$ 推一步去噪，是「预测噪声、换算 $x_0$、重新加噪到上一时刻」的最小实例。
3. **流匹配损失**：换一种建模方式，直接回归速度，对比前两题的 $\epsilon$ 预测。
4. **AdaLN-Zero**：转到网络一侧，看条件信息怎样注入 Transformer。

## 面试追问

1. **扩散模型和流匹配的核心区别是什么？**
   扩散模型沿着一个固定的加噪过程学反向去噪，路径由 $\bar\alpha_t$ 决定，采样是反向去噪的迭代（可随机可确定）；流匹配直接定义噪声到数据的插值路径（常取直线），让网络回归速度场，采样是对 ODE 的积分。数学上同属 $x_t = a_t x_{\text{data}} + b_t \epsilon$ 这一族，差别在路径和参数化。
2. **$\epsilon$、$x_0$、$v$ 预测怎么选？**
   三者可以互相换算，区别是各个噪声水平上的损失权重。$\epsilon$ 预测在低噪声处权重大，但在终端信噪比为 $0$ 时由它恢复 $x_0$ 会除以零；$x_0$ 预测相对 $\epsilon$ 预测更看重高噪声处；$v$ 预测两端都稳，是更常见的折中。
3. **DDIM 为什么比 DDPM 快？**
   DDIM 去掉了每步的随机噪声（$\eta = 0$），采样变成确定性的，相邻两步不必靠得很近，可以只走一个时间步子序列。DDPM 的反向转移需要小步长才近似高斯，所以步数多。
4. **classifier-free guidance 是什么？代价是什么？**
   训练时随机丢弃条件，让同一个网络同时学条件和无条件预测；采样时把两者的差乘以 $w$ 加到无条件预测上。$w$ 越大越贴合条件，但多样性越差，也可能过饱和。代价是每步两次前向。
5. **为什么 DiT 用 AdaLN-Zero 而不是交叉注意力注入时间步和类别？**
   时间步和类别是一个全局向量，不需要在 token 间逐一对齐，用它回归缩放和偏移就够了，比交叉注意力便宜；零初始化的门控还让每个块起点是恒等映射。DiT 论文的对比中它的效果最好。

