## 背景

**静态批处理**：凑齐一批请求一起跑，整批中最长的那个结束后，才换下一批。问题很明显：

- 短请求早就生成完了，却要陪着长请求占着位置，GPU 上一部分算力在空转
- 新来的请求必须等当前整批结束才能开始，延迟很高

**连续批处理**（迭代级调度，Orca 提出，vLLM 采用）：调度的粒度从「一批」变成「一步」。每生成一个 token 就重新调度一次：结束的请求立刻离开，等待的请求立刻补进来。

## 核心概念

**两种步骤的代价不同**：

- **prefill**：新请求第一次运行，一次处理整个提示词（花费 `prompt_len` 个 token），并产生第一个输出 token
- **decode**：之后每步生成一个 token（花费 $1$ 个 token）

**每一步的预算约束**：

$$
\begin{aligned}
\underbrace{|\text{running}|}_{\text{decode 各 1}} + \sum_{r \in \text{本步新准入}} \text{prompt\_len}_r &\le \text{max\_batch\_tokens} \\
|\text{running}| + |\text{新准入}| &\le \text{max\_num\_seqs}
\end{aligned}
$$

**严格先来先服务（FCFS）**：按到达时间（再按编号）排队，队首放不下就停止准入，**不跳过**它去准入后面更小的请求。否则大请求可能永远排不上（饥饿）。

**完成时机**：请求在产生第 `output_len` 个 token 的那一步完成。那一步它仍在批次里（占槽位、占 token），从下一步起才释放。`output_len = 1` 的请求在被准入的那一步（prefill）就完成了。

**示例**：`[(0, 4, 2), (1, 2, 1)]`，预算 8 个 token、最多 2 个序列：

| 步 | decode | 新准入 | token 数 | 本步运行 | 结束 |
| --- | --- | --- | --- | --- | --- |
| 0 | — | 0（prefill 4） | 4 | `[0]` | — |
| 1 | 0（1） | 1（prefill 2） | 3 | `[0, 1]` | 0、1 |

## 用到的 Python 工具

- `sorted(range(n), key=lambda i: (arrival_i, i))`：FCFS 顺序。
- 一个列表记录每个请求还剩几个 token 要生成；`while any(left)` 驱动模拟。

## 常见坑

- **跳过放不下的请求**去准入后面的小请求：违反严格 FCFS。
- **结束的请求过早释放槽位**：它在完成的那一步仍计入 `max_num_seqs` 和 token 预算。
- **准入时忘了算上本步的 decode token**：预算是 decode 和 prefill 共享的。
- **跳过空闲步**：还没有请求到达时，要记录空列表 `[]`。
- **`prompt_len` 超过预算**：这样的请求永远无法调度，要提前抛 `ValueError`，否则会死循环。
- 每一步的请求编号要排好序。
