"""MoE Capacity Dispatch task."""

TASK = {
    "title": "MoE Capacity Dispatch",
    "title_zh": "MoE 容量分发",
    "difficulty": "Hard",
    "description_en": "Implement Mixture-of-Experts token dispatch with a fixed expert capacity (GShard / Switch Transformer).\n\nA router sends each token to its top-k experts, but every expert has a fixed buffer of $C$ slots, so popular experts overflow and the extra assignments are dropped. Under expert parallelism this gather, expert, scatter step is the all-to-all that moves tokens between devices, so the buffer size and the drop rule decide both memory and throughput.\n\n**Signature:** `moe_dispatch(x, router_logits, experts, top_k=2, capacity_factor=1.25) -> tuple[Tensor, int]`\n\n**Parameters:**\n- `x` — token activations (N, D)\n- `router_logits` — router scores (N, E), where $E$ = `len(experts)`\n- `experts` — list of $E$ callables, each mapping (n, D) to (n, D)\n- `top_k` — number of experts per token\n- `capacity_factor` — slack multiplier for the expert buffer\n\n**Returns:** `(out, num_dropped)`. `out` is (N, D) with the same dtype as `x`. `num_dropped` is a Python int: the number of dropped (token, choice) assignments.\n\n**Algorithm:**\n1. `probs = softmax(router_logits, dim=-1)`. Take each token's top_k entries with `torch.topk` (sorted by probability, largest first, so choice rank 0 is the largest). Gate weights are these top-k probabilities renormalized to sum to 1 per token.\n2. Capacity $C = \\left\\lceil \\frac{\\texttt{capacity\\_factor} \\cdot N \\cdot \\texttt{top\\_k}}{E} \\right\\rceil$.\n3. Process assignments in priority order: all tokens' rank-0 choices (token $0, 1, \\ldots, N-1$), then all rank-1 choices, and so on. Each expert accepts assignments until it holds $C$ of them. A later assignment to a full expert is dropped.\n4. Each expert runs once on the stacked rows of its accepted tokens, in acceptance order.\n5. $\\texttt{out}[n]$ = sum over $n$'s accepted assignments of $\\texttt{gate} \\cdot \\texttt{expert\\_output}$. Dropped assignments contribute nothing, and the surviving gates are NOT renormalized.\n\n**Constraints:**\n- An expert that accepted at least one token is called exactly once, with a (count, D) tensor. An expert that accepted none may be skipped, or called once with a (0, D) tensor.\n- Gradients must flow to `x` and `router_logits`.\n- Do not modify the inputs in place.",
    "description_zh": "实现带固定专家容量的混合专家（MoE）token 分发（GShard / Switch Transformer）。\n\n路由器把每个 token 发送到其 top-k 个专家，但每个专家只有固定的 $C$ 个缓冲槽位，热门专家会溢出，多出的分配会被丢弃。在专家并行下，这个 gather、专家计算、scatter 的步骤就是在设备之间搬运 token 的 all-to-all，因此缓冲区大小和丢弃规则同时决定显存与吞吐。\n\n**签名:** `moe_dispatch(x, router_logits, experts, top_k=2, capacity_factor=1.25) -> tuple[Tensor, int]`\n\n**参数:**\n- `x` — token 激活 (N, D)\n- `router_logits` — 路由分数 (N, E)，$E$ = `len(experts)`\n- `experts` — 长度为 $E$ 的可调用对象列表，每个映射 (n, D) 到 (n, D)\n- `top_k` — 每个 token 选择的专家数\n- `capacity_factor` — 专家缓冲区的余量系数\n\n**返回:** `(out, num_dropped)`。`out` 为 (N, D)，dtype 与 `x` 相同。`num_dropped` 为 Python int，即被丢弃的（token, 选择）分配数。\n\n**算法:**\n1. `probs = softmax(router_logits, dim=-1)`。用 `torch.topk` 取每个 token 的前 top_k 个（按概率从大到小排序，选择序号 0 即最大的那个）。门控权重为这 top-k 个概率在每个 token 内重新归一化，使其和为 1。\n2. 容量 $C = \\left\\lceil \\frac{\\texttt{capacity\\_factor} \\cdot N \\cdot \\texttt{top\\_k}}{E} \\right\\rceil$。\n3. 按优先级处理分配：先处理所有 token 的序号 0 选择（token $0, 1, \\ldots, N-1$），再处理所有序号 1 选择，依此类推。每个专家接收分配直到装满 $C$ 个；之后分配给已满专家的请求被丢弃。\n4. 每个专家在其已接收 token 的堆叠行上运行一次，行顺序为接收顺序。\n5. $\\texttt{out}[n]$ = $n$ 的所有已接收分配的 $\\texttt{gate} \\cdot$ 专家输出 之和。被丢弃的分配贡献为 0，剩余门控权重不再重新归一化。\n\n**约束:**\n- 接收了至少一个 token 的专家恰好被调用一次，输入为 (数量, D) 张量。没有接收任何 token 的专家可以不调用，也可以以 (0, D) 张量调用一次。\n- 梯度必须回传到 `x` 和 `router_logits`。\n- 不得原地修改输入。",
    "function_name": "moe_dispatch",
    "hint": "probs = softmax(logits); vals, idx = probs.topk(k); gate = vals / vals.sum(-1)\nC = ceil(cf * N * k / E); fill = [0] * E\nfor r in range(k):              # rank-major priority\n  for n in range(N):\n    e = idx[n, r]\n    if fill[e] < C: fill[e] += 1; accepted[e].append((n, gate[n, r]))\n    else: dropped += 1\nfor e: y = experts[e](x[tokens]); out = out.index_add(0, tokens, gate * y)",
    "hint_zh": "probs = softmax(logits); vals, idx = probs.topk(k); gate = vals / vals.sum(-1)\nC = ceil(cf * N * k / E); fill = [0] * E\nfor r in range(k):              # 按 rank 优先\n  for n in range(N):\n    e = idx[n, r]\n    if fill[e] < C: fill[e] += 1; accepted[e].append((n, gate[n, r]))\n    else: dropped += 1\nfor e: y = experts[e](x[tokens]); out = out.index_add(0, tokens, gate * y)",
    "tests": [
        {
            "name": "Output shape, dtype and drop count type",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
N, D, E = 5, 8, 4
x = torch.randn(N, D)
logits = torch.randn(N, E)
experts = [nn.Linear(D, D) for _ in range(E)]
out, dropped = {fn}(x, logits, experts)
assert out.shape == (N, D), f'Expected ({N}, {D}), got {tuple(out.shape)}'
assert out.dtype == x.dtype, f'Expected dtype {x.dtype}, got {out.dtype}'
assert isinstance(dropped, int) and dropped >= 0, f'num_dropped must be a non-negative int, got {dropped!r}'
""",
        },
        {
            "name": "Large capacity equals dense top-k MoE",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(6)
N, D, E, k = 16, 8, 4, 2
x = torch.randn(N, D)
logits = torch.randn(N, E)
experts = [nn.Linear(D, D) for _ in range(E)]
out, dropped = {fn}(x, logits, experts, top_k=k, capacity_factor=4.0)
assert dropped == 0, f'Large capacity should drop nothing, dropped {dropped}'
with torch.no_grad():
    probs = torch.softmax(logits, dim=-1)
    vals, idx = probs.topk(k, dim=-1)
    gates = vals / vals.sum(dim=-1, keepdim=True)
    ref = torch.zeros(N, D)
    for n in range(N):
        for r in range(k):
            ref[n] += gates[n, r] * experts[int(idx[n, r])](x[n:n + 1])[0]
assert torch.allclose(out, ref, atol=1e-5), f'Max diff vs dense reference: {(out - ref).abs().max().item():.2e}'
""",
        },
        {
            "name": "Capacity is ceil, not floor",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
N, D, E = 6, 4, 4
x = torch.randn(N, D)
logits = torch.zeros(N, E)
logits[:, 0] = 10.0  # every token's only choice is expert 0
experts = [nn.Linear(D, D) for _ in range(E)]
out, dropped = {fn}(x, logits, experts, top_k=1, capacity_factor=1.25)
# C = ceil(1.25 * 6 * 1 / 4) = ceil(1.875) = 2, so tokens 0 and 1 are accepted
assert dropped == 4, f'Expected 4 dropped (C=2), got {dropped}'
with torch.no_grad():
    exp0 = experts[0](x)
assert torch.allclose(out[:2], exp0[:2], atol=1e-5), 'Accepted tokens should get expert 0 output with gate 1.0'
assert torch.all(out[2:] == 0), 'Dropped tokens must produce exactly zero rows'
""",
        },
        {
            "name": "Priority: 1st choices beat a later token's 2nd choice",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
N, D, E = 3, 4, 3
x = torch.randn(N, D)
# top-2 per token: t0 -> (e1, e0), t1 -> (e0, e2), t2 -> (e2, e1)
logits = torch.tensor([[2.0, 3.0, 0.0],
                       [3.0, 0.0, 1.0],
                       [0.0, 1.0, 3.0]])
experts = [nn.Linear(D, D) for _ in range(E)]
# C = ceil(0.5 * 3 * 2 / 3) = 1. Rank-0 choices (e1, e0, e2) all fit.
# Every rank-1 choice hits a full expert, so all three are dropped.
out, dropped = {fn}(x, logits, experts, top_k=2, capacity_factor=0.5)
assert dropped == 3, f'Expected 3 dropped rank-1 assignments, got {dropped}'
p = torch.softmax(logits, dim=-1)
expected = torch.zeros(N, D)
with torch.no_grad():
    for n, e in enumerate([1, 0, 2]):
        top2 = p[n].topk(2).values
        w = top2[0] / top2.sum()  # surviving gate is NOT renormalized to 1
        expected[n] = w * experts[e](x[n:n + 1])[0]
assert torch.allclose(out, expected, atol=1e-5), f'Max diff: {(out - expected).abs().max().item():.2e}'
""",
        },
        {
            "name": "Token with every choice dropped gets a zero row",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(0)
N, D, E, k = 4, 4, 4, 2
x = torch.randn(N, D)
logits = torch.tensor([[3.0, 2.0, 0.0, 0.0]] * N)  # every token: top-2 = (e0, e1)
experts = [nn.Linear(D, D) for _ in range(E)]
# C = ceil(1.0 * 4 * 2 / 4) = 2. Tokens 0 and 1 fill e0 and e1 in both ranks.
# Tokens 2 and 3 lose both of their choices.
out, dropped = {fn}(x, logits, experts, top_k=k, capacity_factor=1.0)
assert dropped == 4, f'Expected 4 dropped assignments, got {dropped}'
assert torch.all(out[2:] == 0), 'Fully dropped tokens must produce exactly zero rows'
p = torch.softmax(logits[0], dim=-1)
w0, w1 = p[0] / (p[0] + p[1]), p[1] / (p[0] + p[1])
with torch.no_grad():
    expected0 = w0 * experts[0](x[0:1])[0] + w1 * experts[1](x[0:1])[0]
assert torch.allclose(out[0], expected0, atol=1e-5), 'Token 0 should combine its two accepted experts'
""",
        },
        {
            "name": "Each expert called once with its accepted rows in order",
            "code": """
import math, torch, torch.nn as nn
torch.manual_seed(5)
N, D, E, k, cf = 12, 4, 4, 2, 0.75
x = torch.randn(N, D)
logits = torch.randn(N, E)
# Independent simulation of the priority rule
C = math.ceil(cf * N * k / E)
top_idx = torch.softmax(logits, dim=-1).topk(k, dim=-1).indices
fill = [0] * E
expected_tokens = [[] for _ in range(E)]
ref_dropped = 0
for r in range(k):
    for n in range(N):
        e = int(top_idx[n, r])
        if fill[e] < C:
            fill[e] += 1
            expected_tokens[e].append(n)
        else:
            ref_dropped += 1

class Counting(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.inputs = []
    def forward(self, h):
        self.inputs.append(h.detach().clone())
        return self.inner(h)

experts = [Counting(nn.Linear(D, D)) for _ in range(E)]
out, dropped = {fn}(x, logits, experts, top_k=k, capacity_factor=cf)
assert dropped == ref_dropped, f'num_dropped: expected {ref_dropped}, got {dropped}'
for e in range(E):
    toks = expected_tokens[e]
    calls = experts[e].inputs
    if toks:
        assert len(calls) == 1, f'expert {e} with {len(toks)} tokens was called {len(calls)} times, expected exactly 1'
        assert torch.equal(calls[0], x[toks]), f'expert {e} got the wrong rows or row order'
    else:
        assert len(calls) == 0 or (len(calls) == 1 and calls[0].shape[0] == 0), f'idle expert {e} should not get rows'
""",
        },
        {
            "name": "Gradients flow to x and router_logits",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(7)
N, D, E = 8, 4, 4
x = torch.randn(N, D, requires_grad=True)
logits = torch.randn(N, E, requires_grad=True)
experts = [nn.Linear(D, D) for _ in range(E)]
out, _ = {fn}(x, logits, experts, top_k=2, capacity_factor=2.0)
out.pow(2).sum().backward()
assert x.grad is not None and x.grad.abs().sum() > 0, 'No gradient reached x'
assert logits.grad is not None and logits.grad.abs().sum() > 0, 'No gradient reached router_logits'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch, torch.nn as nn
torch.manual_seed(8)
N, D, E = 10, 4, 4
x = torch.randn(N, D)
logits = torch.randn(N, E)
x0, l0 = x.clone(), logits.clone()
experts = [nn.Linear(D, D) for _ in range(E)]
out, _ = {fn}(x, logits, experts, top_k=2, capacity_factor=1.0)
assert torch.equal(x, x0), 'x was modified in place'
assert torch.equal(logits, l0), 'router_logits was modified in place'
""",
        },
    ],
    "solution": '''import math


def moe_dispatch(x, router_logits, experts, top_k=2, capacity_factor=1.25):
    N, D = x.shape
    E = router_logits.shape[-1]
    capacity = math.ceil(capacity_factor * N * top_k / E)

    # Top-k gates, renormalized to sum to 1 per token
    probs = torch.softmax(router_logits, dim=-1)               # (N, E)
    top_vals, top_idx = probs.topk(top_k, dim=-1)              # (N, k), sorted desc
    gates = top_vals / top_vals.sum(dim=-1, keepdim=True)      # (N, k)

    # Priority: all rank-0 choices (token order), then all rank-1 choices, ...
    fill = [0] * E
    accepted = [[] for _ in range(E)]   # per expert: list of (token, gate)
    num_dropped = 0
    for r in range(top_k):
        for n in range(N):
            e = int(top_idx[n, r])
            if fill[e] < capacity:
                fill[e] += 1
                accepted[e].append((n, gates[n, r]))
            else:
                num_dropped += 1

    # One batched call per expert; scatter gate-weighted outputs back to tokens
    out = torch.zeros_like(x)
    for e, slots in enumerate(accepted):
        if not slots:
            continue
        tokens = torch.tensor([n for n, _ in slots], device=x.device)
        w = torch.stack([g for _, g in slots])                 # (n_e,)
        y = experts[e](x[tokens])                              # (n_e, D)
        out = out.index_add(0, tokens, w.unsqueeze(-1) * y)
    return out, num_dropped''',
    "demo": """import torch
import torch.nn as nn

torch.manual_seed(0)
N, D, E = 8, 4, 4
x = torch.randn(N, D)
logits = torch.randn(N, E)
experts = [nn.Linear(D, D) for _ in range(E)]

out, dropped = moe_dispatch(x, logits, experts, top_k=2, capacity_factor=1.0)
print('Output shape:', tuple(out.shape), '| dropped assignments (C=1.0):', dropped)

out_big, dropped_big = moe_dispatch(x, logits, experts, top_k=2, capacity_factor=4.0)
print('Output shape:', tuple(out_big.shape), '| dropped assignments (C=4.0):', dropped_big)""",

}
