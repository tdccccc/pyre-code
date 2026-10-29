## 背景

LoRA 只训练少量参数，但冻结的基础权重仍然以 16 位存在显存里。65B 模型光权重就要 130 GB。**QLoRA** 把冻结的基础权重**量化**成 4 位（约 33 GB），LoRA 适配器保持高精度训练。这样单张 48 GB 的 GPU 就能微调 65B 模型。

前向时把量化权重临时**反量化**回浮点参与计算；梯度会穿过反量化后的权重传给输入（以及更早的层），但量化权重本身不更新。

本题用 int8 加逐行缩放来模拟这个过程（真正的 QLoRA 用 4 位的 NF4 格式和双重量化）。

## 核心公式

**逐行对称量化**（见「INT8 量化」一题）：每行一个缩放因子，

$$
s_j = \frac{\max_i |W_{j,i}|}{127},
\qquad
q_{j,i} = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{W_{j,i}}{s_j}\right),\ -127,\ 127\right)
$$

**前向**：

$$
\hat{W} = q \odot s,
\qquad
h = x\, \hat{W}^{\top} + \frac{\alpha}{r}\, x\, A\, B
$$

注意本题 LoRA 矩阵的布局和「LoRA」一题不同：这里 $A \in \mathbb{R}^{d_{\text{in}} \times r}$、$B \in \mathbb{R}^{r \times d_{\text{out}}}$，所以直接写 $x A B$，不用转置。

**初始化**：$B = 0$，$A \sim \mathcal{N}(0, 0.01^2)$。开始时 LoRA 增量为零，输出等于（量化后的）基础层。

## 形状

| 量 | 形状 |
| --- | --- |
| `quantized_weight` | `(out, in)`，`int8` |
| `scale` | `(out, 1)` |
| 反量化权重 $\hat{W}$ | `(out, in)`，`float32` |
| `lora_A` | `(in, rank)` |
| `lora_B` | `(rank, out)` |
| `x`、输出 | `(*, in)`、`(*, out)` |

## 用到的 PyTorch 操作

- `nn.Parameter(t, requires_grad=False)`：整数张量不能求梯度，必须关闭。
- `W.abs().max(dim=1, keepdim=True).values`、`.clamp(min=1e-8)`。
- `t.round().clamp(-127, 127).to(torch.int8)`。
- `param.data.copy_(new)`：在不替换参数对象的前提下写入新值。
- `q.float() * scale`：按行广播反量化。

## 常见坑

- **LoRA 布局照搬 LoRA 一题**：这里的 $A$、$B$ 形状是转置过的，前向是 `x @ A @ B`。
- **对量化权重要求梯度**：int8 张量设置 `requires_grad=True` 会报错。
- **全零行**：最大值为 $0$，缩放因子为 $0$，除以零。要给缩放因子设下限。
- **范围用了 $[-128, 127]$**：本题是对称的 $[-127, 127]$。
- **`set_weight` 里直接重新赋值** `self.quantized_weight = ...`：会替换掉注册好的参数对象，用 `.data.copy_()` 写入。
