## 背景

序列太长时，把激活沿**序列维**切到 $P$ 张卡上（序列并行）。LayerNorm、MLP 等逐 token 的操作各算各的，没有问题；但注意力需要每个 token 看到整条序列。

有两条路线：

- **Ring Attention**：保持序列切分，让 K/V 块沿环传递 $P$ 轮（见「环形注意力」一题）
- **DeepSpeed-Ulysses**：利用「各个头互相独立」这一点，临时把切分方式从「按序列切」换成「按头切」

## 核心公式

**关键观察**：多头注意力中每个头的计算完全独立：

$$
\text{head}_h = \operatorname{softmax}\!\left(\frac{Q_h K_h^{\top}}{\sqrt{D}}\right) V_h
$$

只要某张卡拿到**某些头的完整序列**，就能独立算出这些头的结果。

**两次 all-to-all**（$H$ 个头，$P$ 张卡，要求 $P$ 整除 $H$）：

1. **序列并行 → 头并行**：rank $r$ 原本有全部 $H$ 个头在序列第 $r$ 段的数据。把它按头切成 $P$ 块，第 $s$ 块发给 rank $s$。于是 rank $r$ 收集到所有序列段的第 $r$ 组头（$H/P$ 个），沿序列维拼起来，得到完整序列。
2. **本地注意力**：rank $r$ 对它的 $H/P$ 个头做普通注意力。
3. **头并行 → 序列并行**：反过来，把输出按序列切成 $P$ 段，第 $s$ 段发给 rank $s$；rank $s$ 收集所有头组在第 $s$ 段的输出，沿头维拼接。

**形状变化**（单个 rank）：

$$
\begin{aligned}
(B,\ S/P,\ H,\ D) &\xrightarrow{\text{all-to-all}} (B,\ S,\ H/P,\ D) \\
&\xrightarrow{\text{注意力}} (B,\ S,\ H/P,\ D) \\
&\xrightarrow{\text{all-to-all}} (B,\ S/P,\ H,\ D)
\end{aligned}
$$

**通信**：每次 all-to-all，每个 rank 留下自己的 $1/P$，把其余 $(P-1)/P$ 发出去。

**和 Ring Attention 对比**：

| | Ulysses | Ring Attention |
| --- | --- | --- |
| 通信方式 | 2 次 all-to-all | $P$ 轮点对点传 K/V |
| 并行度上限 | $P \le H$，且 $P$ 整除 $H$ | 不受头数限制 |
| 注意力计算 | 普通注意力，无需改动 | 需要在线 softmax 合并 |

## 用到的 PyTorch 操作

- 切片 `t[:, :, r*h:(r+1)*h]`（按头取块）、`t[:, s*n:(s+1)*n]`（按序列取块）。
- `torch.cat(list, dim=1)`（沿序列拼接）、`torch.cat(list, dim=2)`（沿头拼接）。
- `torch.einsum('bihd,bjhd->bhij', q, k)`、`torch.tril` 构造因果掩码。

## 常见坑

- **头组顺序乱了**：rank $r$ 必须负责第 $r$ 组头，第二次 all-to-all 才能把头按原顺序拼回。
- **因果掩码用了段内位置**：位置 $i$ 能看 $j \le i$ 指的是在完整序列中的**全局**位置。重分片后每个 rank 拥有完整序列，直接用 $S \times S$ 的下三角即可。
- **拼接维度错**：第一次沿序列维（`dim=1`），第二次沿头维（`dim=2`）。
- 缩放因子 $1/\sqrt{D}$，$D$ 是每个头的维度。
- 不要修改输入的分片列表。
