"""Group baselines for critic-free policy gradient: GRPO / Dr. GRPO / RLOO / REINFORCE++ task."""

TASK = {
    "title": "Group Baselines: GRPO / Dr. GRPO / RLOO / REINFORCE++",
    "title_zh": "组内基线：GRPO / Dr. GRPO / RLOO / REINFORCE++",
    "difficulty": "Medium",
    "description_en": "Compute sequence-level advantages for critic-free RL from one scalar reward per response, using four baselines that the RL frameworks offer side by side (verl `adv_estimator`, TRL `scale_rewards`, OpenRLHF `advantage_estimator`).\n\nEach prompt is answered $G$ times; the $G$ responses to one prompt are consecutive entries of `rewards`. Write $\\bar r_g$ for the mean reward of the group that response $i$ belongs to.\n\n**Signature:** `group_advantages(rewards, group_size, method, eps=1e-6) -> Tensor`\n\n**Parameters:**\n- `rewards` — one scalar reward per response (B,), float\n- `group_size` — $G$; $B$ is a multiple of $G$\n- `method` — one of `\"grpo\"`, `\"dr_grpo\"`, `\"rloo\"`, `\"reinforce_pp\"`\n- `eps` — added to the standard deviation\n\n**Returns:** advantages (B,)\n\n**Methods** (std is the **unbiased** standard deviation, as `torch.std`):\n$$\n\\begin{aligned}\n\\texttt{grpo:}\\quad & A_i = \\frac{r_i - \\bar r_g}{\\operatorname{std}(r_g) + \\texttt{eps}} \\\\\n\\texttt{dr\\_grpo:}\\quad & A_i = r_i - \\bar r_g \\\\\n\\texttt{rloo:}\\quad & A_i = r_i - \\frac{1}{G-1}\\sum_{j \\in g,\\, j \\ne i} r_j \\\\\n\\texttt{reinforce\\_pp:}\\quad & x_i = r_i - \\bar r_g, \\qquad A_i = \\frac{x_i - \\operatorname{mean}(x)}{\\operatorname{std}(x) + \\texttt{eps}} \\quad \\text{(mean and std over the whole batch)}\n\\end{aligned}\n$$\n\n**Constraints:**\n- Raise `ValueError` if `group_size < 2`, if $B$ is not a multiple of `group_size`, or for an unknown `method`.\n- The result carries no gradient, even if `rewards` requires grad. Do not modify `rewards`.",
    "description_zh": "无 critic 的强化学习里，每个回答只有一个标量奖励，要从它算出序列级的优势。实现 RL 框架里并列提供的四种基线（verl 的 `adv_estimator`、TRL 的 `scale_rewards`、OpenRLHF 的 `advantage_estimator`）。\n\n每个提示采样 $G$ 个回答，同一提示的 $G$ 个回答在 `rewards` 里是连续的。记 $\\bar r_g$ 为回答 $i$ 所在组的平均奖励。\n\n**签名:** `group_advantages(rewards, group_size, method, eps=1e-6) -> Tensor`\n\n**参数:**\n- `rewards` — 每个回答一个标量奖励 (B,)，浮点\n- `group_size` — $G$；$B$ 是 $G$ 的倍数\n- `method` — `\"grpo\"`、`\"dr_grpo\"`、`\"rloo\"`、`\"reinforce_pp\"` 之一\n- `eps` — 加在标准差上\n\n**返回:** 优势 (B,)\n\n**各方法**（std 均为**无偏**标准差，同 `torch.std`）：\n$$\n\\begin{aligned}\n\\texttt{grpo:}\\quad & A_i = \\frac{r_i - \\bar r_g}{\\operatorname{std}(r_g) + \\texttt{eps}} \\\\\n\\texttt{dr\\_grpo:}\\quad & A_i = r_i - \\bar r_g \\\\\n\\texttt{rloo:}\\quad & A_i = r_i - \\frac{1}{G-1}\\sum_{j \\in g,\\, j \\ne i} r_j \\\\\n\\texttt{reinforce\\_pp:}\\quad & x_i = r_i - \\bar r_g, \\qquad A_i = \\frac{x_i - \\operatorname{mean}(x)}{\\operatorname{std}(x) + \\texttt{eps}} \\quad \\text{（均值和标准差在整个 batch 上算）}\n\\end{aligned}\n$$\n\n**约束:**\n- `group_size < 2`、$B$ 不是 `group_size` 的倍数、`method` 未知时抛出 `ValueError`。\n- 结果不带梯度，即使 `rewards` 需要梯度。不要修改 `rewards`。",
    "function_name": "group_advantages",
    "hint": "r = rewards.detach().view(-1, G)\nmean = r.mean(1, keepdim=True)\ngrpo:    (r - mean) / (r.std(1, keepdim=True) + eps)\ndr_grpo: r - mean\nrloo:    r - (r.sum(1, keepdim=True) - r) / (G - 1)\nreinforce_pp: x = (r - mean).view(-1); (x - x.mean()) / (x.std() + eps)",
    "hint_zh": "r = rewards.detach().view(-1, G)\nmean = r.mean(1, keepdim=True)\ngrpo:    (r - mean) / (r.std(1, keepdim=True) + eps)\ndr_grpo: r - mean\nrloo:    r - (r.sum(1, keepdim=True) - r) / (G - 1)\nreinforce_pp: x = (r - mean).view(-1); (x - x.mean()) / (x.std() + eps)",
    "tests": [
        {
            "name": "Shape (B,), no gradient, input not mutated",
            "code": """
import torch
torch.manual_seed(0)
G, B = 4, 12
rewards = torch.randn(B).requires_grad_(True)
saved = rewards.detach().clone()
for method in ['grpo', 'dr_grpo', 'rloo', 'reinforce_pp']:
    out = {fn}(rewards, G, method)
    assert isinstance(out, torch.Tensor) and out.shape == (B,), f'{method}: expected shape ({B},), got {tuple(out.shape)}'
    assert torch.isfinite(out).all(), f'{method}: advantages must be finite'
    assert not out.requires_grad, f'{method}: advantages must carry no gradient (detach the rewards)'
assert torch.equal(rewards.detach(), saved), 'rewards was modified in place'
""",
        },
        {
            "name": "grpo: per-group z-score with the unbiased std and eps",
            "code": """
import torch
torch.manual_seed(1)
G, P = 3, 4
rewards = torch.randn(G * P) + torch.arange(P).repeat_interleave(G) * 5.0  # groups with very different means
for eps in [1e-6, 0.5]:
    out = {fn}(rewards, G, 'grpo', eps=eps)
    want = torch.empty_like(rewards)
    for p in range(P):
        r = rewards[p * G:(p + 1) * G]
        mu = r.sum() / G
        sd = (((r - mu) ** 2).sum() / (G - 1)).sqrt()
        want[p * G:(p + 1) * G] = (r - mu) / (sd + eps)
    assert torch.allclose(out, want, atol=1e-5), f'eps={eps}: got {out.tolist()}, expected {want.tolist()} (per group, unbiased std, eps added to the std)'
same = {fn}(torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0]), G, 'grpo')
assert torch.isfinite(same).all() and same[:G].abs().max() == 0, 'A group with identical rewards must give exactly 0 advantage, not NaN'
""",
        },
        {
            "name": "dr_grpo: centered only, no scaling",
            "code": """
import torch
torch.manual_seed(2)
G, P = 4, 3
rewards = torch.randn(G * P) * 3 + torch.arange(P).repeat_interleave(G) * 2.0
out = {fn}(rewards, G, 'dr_grpo', eps=0.5)
want = torch.empty_like(rewards)
for p in range(P):
    r = rewards[p * G:(p + 1) * G]
    want[p * G:(p + 1) * G] = r - r.sum() / G
assert torch.allclose(out, want, atol=1e-5), f'got {out.tolist()}, expected r - group mean {want.tolist()}'
""",
        },
        {
            "name": "rloo: leave-one-out baseline",
            "code": """
import torch
torch.manual_seed(3)
for G in [2, 3, 5]:
    P = 3
    rewards = torch.randn(G * P) + torch.arange(P).repeat_interleave(G) * 4.0
    out = {fn}(rewards, G, 'rloo')
    want = torch.empty_like(rewards)
    for p in range(P):
        for j in range(G):
            i = p * G + j
            others = [rewards[p * G + k] for k in range(G) if k != j]
            want[i] = rewards[i] - torch.stack(others).mean()
    assert torch.allclose(out, want, atol=1e-5), f'G={G}: got {out.tolist()}, expected {want.tolist()} (baseline = mean of the OTHER responses in the group)'
# Equivalent closed form: G / (G - 1) * (r - group mean)
r = torch.tensor([1.0, 0.0, 0.0, 0.0])
out = {fn}(r, 4, 'rloo')
assert torch.allclose(out, torch.tensor([1.0, -1/3, -1/3, -1/3]), atol=1e-6), f'got {out.tolist()} for rewards [1, 0, 0, 0]'
""",
        },
        {
            "name": "reinforce_pp: group mean, then whiten over the whole batch",
            "code": """
import torch
torch.manual_seed(4)
G, P = 4, 3
# Group 0 has a tiny spread, group 2 a large one: per-group scaling would blow group 0 up
rewards = torch.cat([torch.randn(G) * 0.01, torch.randn(G), torch.randn(G) * 10]) + torch.arange(P).repeat_interleave(G) * 7.0
for eps in [1e-6, 0.5]:
    out = {fn}(rewards, G, 'reinforce_pp', eps=eps)
    x = torch.empty_like(rewards)
    for p in range(P):
        r = rewards[p * G:(p + 1) * G]
        x[p * G:(p + 1) * G] = r - r.sum() / G
    mu = x.sum() / x.numel()
    sd = (((x - mu) ** 2).sum() / (x.numel() - 1)).sqrt()
    want = (x - mu) / (sd + eps)
    assert torch.allclose(out, want, atol=1e-5), f'eps={eps}: got {out.tolist()}, expected {want.tolist()} (subtract the group mean, then whiten with batch mean and unbiased batch std)'
""",
        },
        {
            "name": "Invalid arguments raise ValueError",
            "code": """
import torch
rewards = torch.randn(6)
for args, why in [((rewards, 3, 'ppo'), 'unknown method'), ((rewards, 1, 'grpo'), 'group_size < 2'),
                  ((rewards, 1, 'rloo'), 'group_size < 2'), ((rewards, 4, 'dr_grpo'), 'B not a multiple of group_size')]:
    try:
        {fn}(*args)
    except ValueError:
        continue
    raise AssertionError(f'Expected ValueError for {why}')
""",
        },
    ],
    "solution": '''def group_advantages(rewards, group_size, method, eps=1e-6):
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if rewards.numel() % group_size != 0:
        raise ValueError("the batch size must be a multiple of group_size")
    G = group_size
    r = rewards.detach().float().view(-1, G)            # (P, G), one row per prompt
    mean = r.mean(dim=1, keepdim=True)

    if method == "grpo":
        adv = (r - mean) / (r.std(dim=1, keepdim=True) + eps)
    elif method == "dr_grpo":
        adv = r - mean
    elif method == "rloo":
        # Baseline for response i: mean of the other G - 1 responses to the same prompt
        adv = r - (r.sum(dim=1, keepdim=True) - r) / (G - 1)
    elif method == "reinforce_pp":
        # Group mean as the baseline, then one global scale for the whole batch
        x = (r - mean).view(-1)
        return (x - x.mean()) / (x.std() + eps)
    else:
        raise ValueError(f"unknown method: {method}")
    return adv.view(-1)''',
    "demo": """import torch

G = 4
# Two prompts: an easy one (3/4 correct) and a hard one (1/4 correct), plus a third with partial credit
rewards = torch.tensor([1.0, 1.0, 1.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.2, 0.3, 0.1, 0.2])
for method in ['grpo', 'dr_grpo', 'rloo', 'reinforce_pp']:
    adv = group_advantages(rewards, G, method)
    print(f'{method:13s}', [round(a, 3) for a in adv.tolist()])
print('grpo scales the low-variance third group up to the size of the others; dr_grpo, rloo and reinforce_pp keep it small')""",

}
