"""RLHF Rewards + GAE task."""

_REF = """
def _ref_gae(scores, kl, values, mask, kl_coef, gamma, lam):
    B, T = values.shape
    adv = torch.zeros(B, T, dtype=torch.float64)
    ret = torch.zeros(B, T, dtype=torch.float64)
    for b in range(B):
        L = int(mask[b].sum())
        r = [-kl_coef * float(kl[b, t]) for t in range(L)]
        r[L - 1] += float(scores[b])
        running = 0.0
        for t in reversed(range(L)):
            next_v = float(values[b, t + 1]) if t + 1 < L else 0.0
            delta = r[t] + gamma * next_v - float(values[b, t])
            running = delta + gamma * lam * running
            adv[b, t] = running
            ret[b, t] = running + float(values[b, t])
    return adv, ret

def _make(B, T, lengths, seed):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(B, T)
    for b, L in enumerate(lengths):
        mask[b, :L] = 1
    scores = torch.randn(B, generator=g)
    # Padding holds garbage on purpose: nothing may be read from it
    kl = torch.randn(B, T, generator=g).abs() * 0.5 + (1 - mask) * 7.0
    values = torch.randn(B, T, generator=g) + (1 - mask) * 5.0
    return scores, kl, values, mask
"""

TASK = {
    "title": "RLHF Rewards + GAE",
    "title_zh": "RLHF 奖励与 GAE",
    "difficulty": "Medium",
    "description_en": "Turn a sequence-level reward into per-token advantages and returns, as PPO-based RLHF does before every update.\n\nThe reward model scores a whole response, but PPO acts on tokens. The standard recipe (InstructGPT, OpenRLHF, verl) gives every token a KL penalty against the reference model, adds the score at the last response token, and then runs Generalized Advantage Estimation (GAE) backwards with the critic's values.\n\n**Signature:** `gae_advantage(scores, kl, values, mask, kl_coef=0.05, gamma=1.0, lam=0.95) -> tuple[Tensor, Tensor]`\n\n**Parameters:**\n- `scores` — sequence-level reward per response (B,)\n- `kl` — per-token KL estimate against the reference model (B, T)\n- `values` — critic values $V_t$ at each response token (B, T)\n- `mask` — response mask (B, T): each row is a run of 1s starting at column 0 followed by 0s (right padding); every row has at least one 1\n- `kl_coef` — KL penalty coefficient $\\beta$\n- `gamma`, `lam` — discount $\\gamma$ and GAE parameter $\\lambda$\n\n**Returns:** `(advantages, returns)`, both (B, T)\n\n**Token rewards** (for a row whose last response token is at index $L-1$):\n$$\nr_t = -\\beta \\cdot \\text{kl}_t + \\mathbb{1}[t = L-1] \\cdot \\text{score}, \\qquad t < L\n$$\n\n**GAE:**\n$$\n\\begin{aligned}\n\\delta_t &= r_t + \\gamma V_{t+1} - V_t, \\qquad V_L := 0 \\text{ (terminal)} \\\\\n\\hat{A}_t &= \\delta_t + \\gamma \\lambda \\hat{A}_{t+1}, \\qquad \\hat{A}_L := 0 \\\\\nR_t &= \\hat{A}_t + V_t\n\\end{aligned}\n$$\n\n**Constraints:**\n- Padding positions ($t \\ge L$) must not influence anything; both outputs are 0 there.\n- Outputs carry no gradient (they are targets, not part of the graph).\n- No advantage whitening here.\n- `mask` may be float or bool. Do not modify the inputs.",
    "description_zh": "把序列级奖励变成逐 token 的优势和回报，这是基于 PPO 的 RLHF 每次更新前都要做的一步。\n\n奖励模型给整条回答打一个分，而 PPO 按 token 行动。标准做法（InstructGPT、OpenRLHF、verl）是：每个 token 都扣一份相对参考模型的 KL 惩罚，在最后一个回答 token 上加上分数，再用 critic 的价值从后往前做广义优势估计（GAE）。\n\n**签名:** `gae_advantage(scores, kl, values, mask, kl_coef=0.05, gamma=1.0, lam=0.95) -> tuple[Tensor, Tensor]`\n\n**参数:**\n- `scores` — 每条回答的序列级奖励 (B,)\n- `kl` — 每个 token 相对参考模型的 KL 估计 (B, T)\n- `values` — critic 在每个回答 token 上的价值 $V_t$ (B, T)\n- `mask` — 回答掩码 (B, T)：每行是从第 0 列开始的一段 1，之后全是 0（右侧填充）；每行至少有一个 1\n- `kl_coef` — KL 惩罚系数 $\\beta$\n- `gamma`、`lam` — 折扣因子 $\\gamma$ 与 GAE 参数 $\\lambda$\n\n**返回:** `(advantages, returns)`，形状都是 (B, T)\n\n**逐 token 奖励**（某行最后一个回答 token 的下标为 $L-1$）：\n$$\nr_t = -\\beta \\cdot \\text{kl}_t + \\mathbb{1}[t = L-1] \\cdot \\text{score}, \\qquad t < L\n$$\n\n**GAE:**\n$$\n\\begin{aligned}\n\\delta_t &= r_t + \\gamma V_{t+1} - V_t, \\qquad V_L := 0 \\text{（终止状态）} \\\\\n\\hat{A}_t &= \\delta_t + \\gamma \\lambda \\hat{A}_{t+1}, \\qquad \\hat{A}_L := 0 \\\\\nR_t &= \\hat{A}_t + V_t\n\\end{aligned}\n$$\n\n**约束:**\n- 填充位置（$t \\ge L$）不得影响任何结果；两个输出在这些位置都为 0。\n- 输出不带梯度（它们是目标值，不属于计算图）。\n- 这里不做优势标准化。\n- `mask` 可以是 float 或 bool。不得修改输入。",
    "function_name": "gae_advantage",
    "hint": "m = mask.float(); L = m.sum(1).long()\nrewards = -kl_coef * kl * m; rewards[arange(B), L - 1] += scores\nnext_v[:, :-1] = values[:, 1:] * m[:, 1:]   # 0 after the last token\ndelta = rewards + gamma * next_v - values * m\nfor t from T-1 down to 0: A = delta[:, t] + gamma * lam * A; adv[:, t] = A\nreturns = (adv + values) * m   # all under no_grad",
    "hint_zh": "m = mask.float(); L = m.sum(1).long()\nrewards = -kl_coef * kl * m; rewards[arange(B), L - 1] += scores\nnext_v[:, :-1] = values[:, 1:] * m[:, 1:]   # 最后一个 token 之后为 0\ndelta = rewards + gamma * next_v - values * m\nt 从 T-1 倒序到 0：A = delta[:, t] + gamma * lam * A；adv[:, t] = A\nreturns = (adv + values) * m   # 全部在 no_grad 下",
    "tests": [
        {
            "name": "Shapes, zeros on padding, no gradient",
            "code": _REF + """
scores, kl, values, mask = _make(3, 6, [6, 4, 1], seed=0)
values = values.requires_grad_(True)
adv, ret = {fn}(scores, kl, values, mask)
assert adv.shape == (3, 6) and ret.shape == (3, 6), f'Expected (3, 6) outputs, got {tuple(adv.shape)} and {tuple(ret.shape)}'
pad = mask == 0
assert torch.all(adv[pad] == 0), 'Advantages must be 0 on padding'
assert torch.all(ret[pad] == 0), 'Returns must be 0 on padding'
assert not adv.requires_grad and not ret.requires_grad, 'Advantages and returns must not carry gradient'
""",
        },
        {
            "name": "Matches a step-by-step GAE reference (variable lengths)",
            "code": _REF + """
scores, kl, values, mask = _make(4, 7, [7, 3, 5, 1], seed=1)
for kl_coef, gamma, lam in [(0.05, 1.0, 0.95), (0.1, 0.9, 0.8)]:
    adv, ret = {fn}(scores, kl, values, mask, kl_coef=kl_coef, gamma=gamma, lam=lam)
    ref_adv, ref_ret = _ref_gae(scores, kl, values, mask, kl_coef, gamma, lam)
    assert torch.allclose(adv.double(), ref_adv, atol=1e-5), f'(beta={kl_coef}, gamma={gamma}, lam={lam}) advantages max diff {(adv.double() - ref_adv).abs().max().item():.2e}'
    assert torch.allclose(ret.double(), ref_ret, atol=1e-5), f'(beta={kl_coef}, gamma={gamma}, lam={lam}) returns max diff {(ret.double() - ref_ret).abs().max().item():.2e}'
""",
        },
        {
            "name": "Score lands on the last response token, not the last column",
            "code": _REF + """
mask = torch.tensor([[1., 1., 1., 0., 0.]])
kl = torch.zeros(1, 5)
values = torch.zeros(1, 5)
scores = torch.tensor([2.0])
adv, ret = {fn}(scores, kl, values, mask, gamma=1.0, lam=1.0)
# With zero values and zero KL, every response token's return is the final score
assert torch.allclose(ret, torch.tensor([[2., 2., 2., 0., 0.]]), atol=1e-6), f'Got returns {ret.tolist()}'
""",
        },
        {
            "name": "lam=1, gamma=1: returns are the reward-to-go",
            "code": _REF + """
scores, kl, values, mask = _make(3, 6, [6, 2, 4], seed=2)
adv, ret = {fn}(scores, kl, values, mask, kl_coef=0.2, gamma=1.0, lam=1.0)
m = mask
r = -0.2 * kl * m
L = m.sum(1).long()
r[torch.arange(3), L - 1] += scores
togo = torch.flip(torch.cumsum(torch.flip(r, [1]), 1), [1]) * m
assert torch.allclose(ret, togo, atol=1e-5), f'With lam=1 the returns should equal the reward-to-go; max diff {(ret - togo).abs().max().item():.2e}'
""",
        },
        {
            "name": "lam=0: advantage is the one-step TD error",
            "code": _REF + """
scores, kl, values, mask = _make(2, 5, [5, 3], seed=3)
gamma = 0.95
adv, _ = {fn}(scores, kl, values, mask, kl_coef=0.1, gamma=gamma, lam=0.0)
m = mask
r = -0.1 * kl * m
L = m.sum(1).long()
r[torch.arange(2), L - 1] += scores
next_v = torch.zeros_like(values)
next_v[:, :-1] = values[:, 1:] * m[:, 1:]
td = (r + gamma * next_v - values) * m
assert torch.allclose(adv, td, atol=1e-5), f'With lam=0 the advantage is delta_t; max diff {(adv - td).abs().max().item():.2e}'
""",
        },
        {
            "name": "Bool mask accepted; inputs not mutated",
            "code": _REF + """
scores, kl, values, mask = _make(3, 5, [5, 2, 3], seed=4)
s0, k0, v0, m0 = scores.clone(), kl.clone(), values.clone(), mask.clone()
adv_f, ret_f = {fn}(scores, kl, values, mask)
assert torch.equal(scores, s0) and torch.equal(kl, k0) and torch.equal(values, v0) and torch.equal(mask, m0), 'Inputs were modified in place'
adv_b, ret_b = {fn}(scores, kl, values, mask.bool())
assert torch.allclose(adv_f, adv_b) and torch.allclose(ret_f, ret_b), 'Bool and float masks must give the same result'
""",
        },
    ],
    "solution": '''def gae_advantage(scores, kl, values, mask, kl_coef=0.05, gamma=1.0, lam=0.95):
    with torch.no_grad():
        m = mask.to(values.dtype)
        B, T = values.shape
        lengths = m.sum(dim=1).long()
        # Per-token reward: KL penalty everywhere, sequence score on the last response token
        rewards = -kl_coef * kl * m
        rewards[torch.arange(B), lengths - 1] += scores
        # V_{t+1}, with 0 after the last response token (terminal state, never read padding)
        next_values = torch.zeros_like(values)
        next_values[:, :-1] = values[:, 1:] * m[:, 1:]
        deltas = rewards + gamma * next_values - values * m
        advantages = torch.zeros_like(values)
        running = torch.zeros(B, dtype=values.dtype)
        for t in reversed(range(T)):
            # Padding has delta = 0 and comes after the response, so running stays 0 there
            running = deltas[:, t] + gamma * lam * running
            advantages[:, t] = running
        advantages = advantages * m
        returns = (advantages + values) * m
    return advantages, returns''',
    "demo": """import torch

torch.manual_seed(0)
mask = torch.tensor([[1., 1., 1., 1., 1.],
                     [1., 1., 1., 0., 0.]])
scores = torch.tensor([1.0, -0.5])
kl = torch.rand(2, 5) * 0.2 * mask
values = torch.randn(2, 5) * 0.3 * mask

adv, ret = gae_advantage(scores, kl, values, mask, kl_coef=0.05, gamma=1.0, lam=0.95)
print('advantages:\\n', adv.round(decimals=3))
print('returns:\\n', ret.round(decimals=3))""",

}
