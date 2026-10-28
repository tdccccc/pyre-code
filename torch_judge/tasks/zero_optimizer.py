"""ZeRO-2 Sharded Adam task."""

TASK = {
    "title": "ZeRO-2 Sharded Adam",
    "title_zh": "ZeRO-2 分片 Adam",
    "difficulty": "Hard",
    "description_en": "Implement one ZeRO stage-2 sharded Adam step (single-process simulation of data-parallel training).\n\nIn plain data parallelism every rank keeps a full copy of the parameters, the gradients and the Adam moments. ZeRO-2 shards the gradients and the optimizer states: rank $r$ owns only partition $r$ of the flat parameter vector and keeps the Adam moments for that partition alone. A step reduce-scatters the gradients so each rank receives the averaged gradient of its own partition, updates only that partition, and then all-gathers the updated partitions so every rank has the full parameters again. The result is the same as ordinary Adam on the averaged gradient, but each rank stores only $1/W$ of the optimizer memory.\n\n**Signature:** `zero2_adam_step(flat_param, rank_grads, opt_states, step, lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -> tuple[Tensor, list[dict]]`\n\n**Parameters:**\n- `flat_param` — 1-D float tensor of length $N$: the full parameters, identical on every rank\n- `rank_grads` — list of `W = len(rank_grads)` full-length gradients of shape `(N,)`, one per rank, each computed on that rank's own data\n- `opt_states` — list of $W$ dicts `{\"m\": Tensor, \"v\": Tensor}`. Entry $r$ holds the Adam first and second moments of partition $r$, each of shape `(N // W,)`\n- `step` — int $\\ge 1$, the Adam step number used for bias correction\n- `lr`, `betas`, `eps` — Adam hyperparameters\n\n**Returns:** `(new_flat_param, new_opt_states)` — the new full parameter tensor of shape `(N,)`, and a list of $W$ new state dicts with the same layout as `opt_states`.\n\n**Algorithm:** (partition $r$ is the elements $[r \\cdot N/W,\\ (r+1) \\cdot N/W)$; $g$ is the gradient restricted to partition $r$):\n**Step 1, reduce-scatter:** $g = \\frac{1}{W} \\sum_k \\texttt{rank\\_grads}[k][\\text{partition } r]$ (average, not sum)\n\n**Step 2, local Adam** on partition $r$, with $(\\beta_1, \\beta_2)$ = `betas` and $\\epsilon$ = `eps`:\n\n$$\n\\begin{aligned}\nm &\\leftarrow \\beta_1 m + (1-\\beta_1)\\, g \\\\\nv &\\leftarrow \\beta_2 v + (1-\\beta_2)\\, g^2 \\\\\n\\hat{m} &= m / (1-\\beta_1^{\\texttt{step}}) \\\\\n\\hat{v} &= v / (1-\\beta_2^{\\texttt{step}}) \\\\\np_r &\\leftarrow p_r - \\texttt{lr} \\cdot \\frac{\\hat{m}}{\\sqrt{\\hat{v}} + \\epsilon}\n\\end{aligned}\n$$\n\n**Step 3, all-gather:** `new_flat_param` is the concatenation of $p_r$ for $r = 0, \\dots, W-1$\n\n**Constraints:**\n- $N$ is divisible by $W$. $W = 1$ is valid and reduces to plain Adam.\n- Must match `torch.optim.Adam` (no weight decay, no amsgrad) run on the averaged gradient, including over several consecutive steps when the returned states are fed back in.\n- Do not mutate `flat_param`, `rank_grads`, or the tensors in `opt_states`. Return new tensors.",
    "description_zh": "实现一步 ZeRO stage-2 分片 Adam 更新（单进程模拟数据并行训练）。\n\n在普通数据并行中，每个 rank 都保存完整的参数、梯度和 Adam 动量。ZeRO-2 对梯度和优化器状态进行分片：rank $r$ 只负责扁平参数向量的第 $r$ 个分区，并且只保存该分区的 Adam 动量。一步更新中，先对梯度做 reduce-scatter，使每个 rank 拿到自己分区的平均梯度；然后只更新本分区；最后 all-gather 各分区更新后的参数，使每个 rank 再次拥有完整参数。结果与在平均梯度上运行普通 Adam 完全相同，但每个 rank 只需存储 $1/W$ 的优化器内存。\n\n**签名:** `zero2_adam_step(flat_param, rank_grads, opt_states, step, lr=1e-3, betas=(0.9, 0.999), eps=1e-8) -> tuple[Tensor, list[dict]]`\n\n**参数:**\n- `flat_param` — 长度为 $N$ 的一维浮点张量：完整参数，每个 rank 上相同\n- `rank_grads` — 长度为 `W = len(rank_grads)` 的列表，每个元素是形状 `(N,)` 的完整梯度，每个 rank 一个，由该 rank 自己的数据计算得到\n- `opt_states` — 长度为 $W$ 的字典列表，每个字典为 `{\"m\": Tensor, \"v\": Tensor}`。第 $r$ 项保存分区 $r$ 的 Adam 一阶和二阶动量，形状均为 `(N // W,)`\n- `step` — 整数 $\\ge 1$，用于偏差修正的 Adam 步数\n- `lr`、`betas`、`eps` — Adam 超参数\n\n**返回:** `(new_flat_param, new_opt_states)` —— 形状为 `(N,)` 的新完整参数张量，以及长度为 $W$ 的新状态字典列表，其布局与 `opt_states` 相同。\n\n**算法:**（分区 $r$ 为元素 $[r \\cdot N/W,\\ (r+1) \\cdot N/W)$；$g$ 为限制在分区 $r$ 上的梯度）\n**第 1 步 reduce-scatter：** $g = \\frac{1}{W} \\sum_k \\texttt{rank\\_grads}[k][\\text{分区 } r]$（取平均，不是求和）\n\n**第 2 步 本地 Adam**，作用于分区 $r$，其中 $(\\beta_1, \\beta_2)$ = `betas`，$\\epsilon$ = `eps`：\n\n$$\n\\begin{aligned}\nm &\\leftarrow \\beta_1 m + (1-\\beta_1)\\, g \\\\\nv &\\leftarrow \\beta_2 v + (1-\\beta_2)\\, g^2 \\\\\n\\hat{m} &= m / (1-\\beta_1^{\\texttt{step}}) \\\\\n\\hat{v} &= v / (1-\\beta_2^{\\texttt{step}}) \\\\\np_r &\\leftarrow p_r - \\texttt{lr} \\cdot \\frac{\\hat{m}}{\\sqrt{\\hat{v}} + \\epsilon}\n\\end{aligned}\n$$\n\n**第 3 步 all-gather：** `new_flat_param` 为 $r = 0, \\dots, W-1$ 的 $p_r$ 依次拼接\n\n**约束:**\n- $N$ 可被 $W$ 整除。$W = 1$ 也是合法输入，此时退化为普通 Adam。\n- 必须与在平均梯度上运行的 `torch.optim.Adam`（无 weight decay，无 amsgrad）一致；连续多步时，将返回的状态再次传入，结果也必须一致。\n- 不得修改 `flat_param`、`rank_grads` 或 `opt_states` 中的张量。请返回新的张量。",
    "function_name": "zero2_adam_step",
    "hint": "n = N // W;  b1, b2 = betas\nfor r in range(W):                       # each rank, its own partition\n    lo, hi = r*n, (r+1)*n\n    g = mean(rank_grads[k][lo:hi] for k)  # reduce-scatter: average\n    m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g*g\n    p_r = flat[lo:hi] - lr * (m/(1-b1^step)) / (sqrt(v/(1-b2^step)) + eps)\nnew_flat = cat(p_r over r)               # all-gather",
    "hint_zh": "n = N // W;  b1, b2 = betas\nfor r in range(W):                       # 每个 rank 处理自己的分区\n    lo, hi = r*n, (r+1)*n\n    g = mean(rank_grads[k][lo:hi] for k)  # reduce-scatter：取平均\n    m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g*g\n    p_r = flat[lo:hi] - lr * (m/(1-b1^step)) / (sqrt(v/(1-b2^step)) + eps)\nnew_flat = cat(p_r over r)               # all-gather",
    "tests": [
        {
            "name": "State and output shapes",
            "code": """
import torch
torch.manual_seed(0)
N, W = 12, 4
flat = torch.randn(N)
grads = [torch.randn(N) for _ in range(W)]
states = [{"m": torch.zeros(N // W), "v": torch.zeros(N // W)} for _ in range(W)]
new_flat, new_states = {fn}(flat, grads, states, step=1)
assert new_flat.shape == (N,), f'Expected full param shape ({N},), got {tuple(new_flat.shape)}'
assert len(new_states) == W, f'Expected {W} state dicts, got {len(new_states)}'
for r, s in enumerate(new_states):
    assert set(s.keys()) == {"m", "v"}, f'Rank {r}: state keys {set(s.keys())}'
    assert s["m"].shape == (N // W,), f'Rank {r}: m shape {tuple(s["m"].shape)}'
    assert s["v"].shape == (N // W,), f'Rank {r}: v shape {tuple(s["v"].shape)}'
""",
        },
        {
            "name": "One step matches torch.optim.Adam on the averaged gradient",
            "code": """
import torch
torch.manual_seed(1)
N, W = 12, 2
flat = torch.randn(N)
grads = [torch.randn(N) for _ in range(W)]
states = [{"m": torch.zeros(N // W), "v": torch.zeros(N // W)} for _ in range(W)]
new_flat, _ = {fn}(flat, grads, states, step=1)

avg = torch.stack(grads).mean(dim=0)
p = torch.nn.Parameter(flat.clone())
opt = torch.optim.Adam([p], lr=1e-3)
p.grad = avg.clone()
opt.step()
assert torch.allclose(new_flat, p.detach(), atol=1e-6), 'One ZeRO-2 step should match torch.optim.Adam on the averaged gradient'
""",
        },
        {
            "name": "Five consecutive steps match torch.optim.Adam (W=1, 2, 4)",
            "code": """
import torch
for W in [1, 2, 4]:
    torch.manual_seed(10 + W)
    N = 12
    flat = torch.randn(N)
    states = [{"m": torch.zeros(N // W), "v": torch.zeros(N // W)} for _ in range(W)]
    p = torch.nn.Parameter(flat.clone())
    opt = torch.optim.Adam([p], lr=1e-3)
    for step in range(1, 6):
        grads = [torch.randn(N) for _ in range(W)]
        flat, states = {fn}(flat, grads, states, step=step)
        p.grad = torch.stack(grads).mean(dim=0)
        opt.step()
        assert torch.allclose(flat, p.detach(), atol=1e-6), f'W={W}, step {step}: diverged from torch.optim.Adam'
""",
        },
        {
            "name": "Averages gradients across ranks (not sums)",
            "code": """
import torch
torch.manual_seed(2)
N, W = 6, 3
flat = torch.randn(N)
grads = [torch.randn(N) for _ in range(W)]
states = [{"m": torch.zeros(N // W), "v": torch.zeros(N // W)} for _ in range(W)]
_, new_states = {fn}(flat, grads, states, step=1)
avg = torch.stack(grads).mean(dim=0)
n = N // W
b1 = 0.9
for r in range(W):
    expected_m = (1 - b1) * avg[r * n:(r + 1) * n]
    assert torch.allclose(new_states[r]["m"], expected_m, atol=1e-6), \\
        f'Rank {r}: first moment must come from the averaged gradient (got a sum or wrong slice?)'
""",
        },
        {
            "name": "Bias correction uses the given step",
            "code": """
import torch
torch.manual_seed(3)
N, W = 8, 2
flat = torch.randn(N)
grads = [torch.randn(N) for _ in range(W)]
states = [{"m": torch.zeros(N // W), "v": torch.zeros(N // W)} for _ in range(W)]
step, lr, b1, b2, eps = 4, 1e-3, 0.9, 0.999, 1e-8
new_flat, _ = {fn}(flat, grads, states, step=step)

g = torch.stack(grads).mean(dim=0)
m = (1 - b1) * g
v = (1 - b2) * g * g
m_hat = m / (1 - b1 ** step)
v_hat = v / (1 - b2 ** step)
expected = flat - lr * m_hat / (v_hat.sqrt() + eps)
assert torch.allclose(new_flat, expected, atol=1e-6), 'Bias correction must use the step argument'
""",
        },
        {
            "name": "Custom hyperparameters match torch.optim.Adam",
            "code": """
import torch
torch.manual_seed(4)
N, W = 6, 3
lr, betas, eps = 0.05, (0.5, 0.9), 0.5
flat = torch.randn(N)
states = [{"m": torch.zeros(N // W), "v": torch.zeros(N // W)} for _ in range(W)]
p = torch.nn.Parameter(flat.clone())
opt = torch.optim.Adam([p], lr=lr, betas=betas, eps=eps)
for step in range(1, 4):
    grads = [torch.randn(N) for _ in range(W)]
    flat, states = {fn}(flat, grads, states, step=step, lr=lr, betas=betas, eps=eps)
    p.grad = torch.stack(grads).mean(dim=0)
    opt.step()
    assert torch.allclose(flat, p.detach(), atol=1e-6), f'Step {step}: custom lr/betas/eps not matched'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(5)
N, W = 12, 4
flat = torch.randn(N)
grads = [torch.randn(N) for _ in range(W)]
states = [{"m": torch.rand(N // W), "v": torch.rand(N // W)} for _ in range(W)]
flat_before = flat.clone()
grads_before = [g.clone() for g in grads]
states_before = [{k: t.clone() for k, t in s.items()} for s in states]
{fn}(flat, grads, states, step=2)
assert torch.equal(flat, flat_before), 'flat_param was modified'
for r in range(W):
    assert torch.equal(grads[r], grads_before[r]), f'rank_grads[{r}] was modified'
    for k in ("m", "v"):
        assert torch.equal(states[r][k], states_before[r][k]), f'opt_states[{r}]["{k}"] was modified'
""",
        },
    ],
    "solution": '''def zero2_adam_step(flat_param, rank_grads, opt_states, step, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    W = len(rank_grads)
    n = flat_param.numel() // W  # partition size
    b1, b2 = betas

    new_parts, new_states = [], []
    for r in range(W):
        lo, hi = r * n, (r + 1) * n

        # 1. Reduce-scatter: rank r receives the average gradient of its partition
        g = torch.stack([grad[lo:hi] for grad in rank_grads]).mean(dim=0)

        # 2. Local Adam update on this partition only
        m = b1 * opt_states[r]["m"] + (1 - b1) * g
        v = b2 * opt_states[r]["v"] + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** step)
        v_hat = v / (1 - b2 ** step)
        new_parts.append(flat_param[lo:hi] - lr * m_hat / (v_hat.sqrt() + eps))
        new_states.append({"m": m, "v": v})

    # 3. All-gather: every rank reassembles the full updated parameter
    new_flat = torch.cat(new_parts)
    return new_flat, new_states''',
    "demo": """torch.manual_seed(0)
N, W = 8, 4
flat = torch.randn(N)
states = [{"m": torch.zeros(N // W), "v": torch.zeros(N // W)} for _ in range(W)]

p = torch.nn.Parameter(flat.clone())
opt = torch.optim.Adam([p], lr=1e-3)
for step in range(1, 4):
    grads = [torch.randn(N) for _ in range(W)]  # per-rank gradients
    flat, states = zero2_adam_step(flat, grads, states, step)
    p.grad = torch.stack(grads).mean(dim=0)
    opt.step()

print("Full param shape:", flat.shape)                       # expect (8,)
print("Moments per rank:", [s["m"].numel() for s in states])  # expect [2, 2, 2, 2]
print("Max diff vs torch Adam:", (flat - p.detach()).abs().max().item())  # expect ~0""",
}
