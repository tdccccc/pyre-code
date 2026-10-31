"""DAPO loss task: dynamic sampling, overlong shaping, clip-higher, token-mean."""

TASK = {
    "title": "DAPO Loss (Clip-Higher, Dynamic Sampling, Overlong Shaping)",
    "title_zh": "DAPO 损失",
    "difficulty": "Hard",
    "description_en": "Implement the policy loss of DAPO (ByteDance Seed, 2025), which changes GRPO in four places to make long chain-of-thought RL stable: drop groups that carry no signal, penalize overlong answers smoothly, raise only the upper clip bound, and average over all tokens in the batch. DAPO has no KL term.\n\n**Signature:** `dapo_loss(logps, old_logps, rewards, lengths, mask, group_size, max_len, cache_len, eps_low=0.2, eps_high=0.28, eps=1e-6) -> Tensor`\n\n**Parameters:**\n- `logps` — current policy log-probs of the sampled tokens (B, T)\n- `old_logps` — log-probs under the policy that generated the batch (B, T), constant\n- `rewards` — raw reward per response (B,), e.g. 1 for correct and 0 (or -1) for wrong, constant\n- `lengths` — response length in tokens (B,), int; used only for the overlong penalty and may differ from the padded width $T$\n- `mask` — (B, T), 1 on response tokens, 0 on padding (float or bool); every row has at least one token\n- `group_size` — $G \\ge 2$; $B$ is a multiple of $G$ and the $G$ responses to one prompt are consecutive rows\n- `max_len` — $L_{\\max}$; `cache_len` — $L_{\\text{cache}}$, the width of the soft-penalty band ($0 < L_{\\text{cache}} \\le L_{\\max}$)\n- `eps_low`, `eps_high` — lower and upper clip ranges; `eps` — added to the std\n\n**Returns:** scalar loss\n\n**Steps:**\n1. **Dynamic sampling.** Drop every group whose **raw** rewards are all equal (all correct or all wrong: zero advantage, no signal). Later steps use only the kept rows.\n2. **Overlong shaping.** With $S = L_{\\max} - L_{\\text{cache}}$:\n$$\nR_{\\text{len}}(L) = \\begin{cases} 0 & L \\le S \\\\ \\dfrac{S - L}{L_{\\text{cache}}} & S < L \\le L_{\\max} \\\\ -1 & L > L_{\\max} \\end{cases}\n\\qquad \\tilde r_i = r_i + R_{\\text{len}}(\\text{lengths}_i)\n$$\n3. **Advantage.** $A_i = \\dfrac{\\tilde r_i - \\operatorname{mean}(\\tilde r_{g(i)})}{\\operatorname{std}(\\tilde r_{g(i)}) + \\texttt{eps}}$ within each kept group, unbiased std (as `torch.std`), shared by all tokens of response $i$.\n4. **Clip-higher.** $\\rho_{i,t} = \\exp(\\text{logps}_{i,t} - \\text{old\\_logps}_{i,t})$, $\\ \\text{surr}_{i,t} = \\min\\bigl(\\rho_{i,t} A_i,\\ \\operatorname{clip}(\\rho_{i,t}, 1-\\epsilon_{\\text{low}}, 1+\\epsilon_{\\text{high}}) A_i\\bigr)$.\n5. **Token-mean.** $L = -\\dfrac{\\sum_{i \\in \\text{kept},\\,t} m_{i,t}\\,\\text{surr}_{i,t}}{\\sum_{i \\in \\text{kept},\\,t} m_{i,t}}$.\n\n**Constraints:**\n- If no group is kept, return a scalar 0.0 that is still connected to `logps`: after `backward()` its gradient is all zeros, not `None`.\n- Gradients flow only into `logps`. Padding positions hold arbitrary finite values and must not affect the result. Do not modify the inputs.",
    "description_zh": "实现 DAPO（字节 Seed，2025）的策略损失。它为了让长思维链的 RL 训练稳定，在四处修改了 GRPO：丢掉没有学习信号的组，对过长的回答做平滑惩罚，只放宽裁剪上界，以及在整个 batch 的所有 token 上取平均。DAPO 没有 KL 项。\n\n**签名:** `dapo_loss(logps, old_logps, rewards, lengths, mask, group_size, max_len, cache_len, eps_low=0.2, eps_high=0.28, eps=1e-6) -> Tensor`\n\n**参数:**\n- `logps` — 当前策略对采样 token 的对数概率 (B, T)\n- `old_logps` — 生成这批数据的策略给出的对数概率 (B, T)，常数\n- `rewards` — 每个回答的原始奖励 (B,)，例如答对 1、答错 0（或 -1），常数\n- `lengths` — 回答的 token 长度 (B,)，整数；只用于超长惩罚，可以和 padding 后的宽度 $T$ 不同\n- `mask` — (B, T)，回答 token 为 1，padding 为 0（float 或 bool）；每行至少有一个 token\n- `group_size` — $G \\ge 2$；$B$ 是 $G$ 的倍数，同一提示的 $G$ 个回答占连续的行\n- `max_len` — $L_{\\max}$；`cache_len` — $L_{\\text{cache}}$，软惩罚区间的宽度（$0 < L_{\\text{cache}} \\le L_{\\max}$）\n- `eps_low`、`eps_high` — 裁剪的下界和上界幅度；`eps` — 加在标准差上\n\n**返回:** 标量损失\n\n**步骤:**\n1. **动态采样。** 丢掉**原始**奖励全部相同的组（全对或全错：优势为 0，没有信号）。之后的步骤只使用保留下来的行。\n2. **超长惩罚。** 记 $S = L_{\\max} - L_{\\text{cache}}$：\n$$\nR_{\\text{len}}(L) = \\begin{cases} 0 & L \\le S \\\\ \\dfrac{S - L}{L_{\\text{cache}}} & S < L \\le L_{\\max} \\\\ -1 & L > L_{\\max} \\end{cases}\n\\qquad \\tilde r_i = r_i + R_{\\text{len}}(\\text{lengths}_i)\n$$\n3. **优势。** 在每个保留的组内 $A_i = \\dfrac{\\tilde r_i - \\operatorname{mean}(\\tilde r_{g(i)})}{\\operatorname{std}(\\tilde r_{g(i)}) + \\texttt{eps}}$，标准差用无偏估计（同 `torch.std`），回答 $i$ 的所有 token 共用。\n4. **Clip-higher。** $\\rho_{i,t} = \\exp(\\text{logps}_{i,t} - \\text{old\\_logps}_{i,t})$，$\\ \\text{surr}_{i,t} = \\min\\bigl(\\rho_{i,t} A_i,\\ \\operatorname{clip}(\\rho_{i,t}, 1-\\epsilon_{\\text{low}}, 1+\\epsilon_{\\text{high}}) A_i\\bigr)$。\n5. **Token 平均。** $L = -\\dfrac{\\sum_{i \\in \\text{保留},\\,t} m_{i,t}\\,\\text{surr}_{i,t}}{\\sum_{i \\in \\text{保留},\\,t} m_{i,t}}$。\n\n**约束:**\n- 如果没有任何组被保留，返回值为 0.0 的标量，并且它仍与 `logps` 相连：`backward()` 后梯度全为 0，而不是 `None`。\n- 梯度只流向 `logps`。padding 位置可以是任意有限值，不得影响结果。不要修改输入。",
    "function_name": "dapo_loss",
    "hint": "r = rewards.view(-1, G);  keep = r.max(1).values > r.min(1).values      # (P,)\nif not keep.any(): return (logps * 0).sum()\nS = max_len - cache_len;  L = lengths.float()\npen = where(L <= S, 0, where(L <= max_len, (S - L) / cache_len, -1))\nrt = (rewards + pen).view(-1, G);  A = (rt - rt.mean(1, keepdim)) / (rt.std(1, keepdim) + eps)\nm = mask * keep.repeat_interleave(G)[:, None]\nsurr = min(ratio * A, clamp(ratio, 1 - eps_low, 1 + eps_high) * A)\nloss = -(surr * m).sum() / m.sum()",
    "hint_zh": "r = rewards.view(-1, G);  keep = r.max(1).values > r.min(1).values      # (P,)\nif not keep.any(): return (logps * 0).sum()\nS = max_len - cache_len;  L = lengths.float()\npen = where(L <= S, 0, where(L <= max_len, (S - L) / cache_len, -1))\nrt = (rewards + pen).view(-1, G);  A = (rt - rt.mean(1, keepdim)) / (rt.std(1, keepdim) + eps)\nm = mask * keep.repeat_interleave(G)[:, None]\nsurr = min(ratio * A, clamp(ratio, 1 - eps_low, 1 + eps_high) * A)\nloss = -(surr * m).sum() / m.sum()",
    "tests": [
        {
            "name": "Returns a finite scalar with a gradient",
            "code": """
import torch
torch.manual_seed(0)
B, T, G = 4, 5, 2
old = torch.randn(B, T) * 0.3 - 1
logps = (old + 0.1 * torch.randn(B, T)).requires_grad_(True)
rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
lengths = torch.tensor([5, 3, 4, 2])
out = {fn}(logps, old, rewards, lengths, torch.ones(B, T), G, max_len=8, cache_len=4)
assert isinstance(out, torch.Tensor) and out.dim() == 0, 'Loss must be a scalar tensor'
assert torch.isfinite(out), 'Loss should be finite'
assert out.requires_grad, 'Loss must carry a gradient'
""",
        },
        {
            "name": "Matches a reference (filtering, shaping, asymmetric clip, padding)",
            "code": """
import torch
torch.manual_seed(1)
G, P, T = 4, 4, 6
B = G * P
max_len, cache_len = 20, 8            # soft band is (12, 20]
eps_low, eps_high, eps = 0.15, 0.35, 0.1
rewards = torch.tensor([1., 0., 1., 0.,    # mixed: kept
                        1., 1., 1., 1.,    # all correct: dropped
                        0., 1., 0., 0.,    # mixed: kept
                        0., 0., 0., 0.])   # all wrong: dropped even though lengths differ
lengths = torch.tensor([5, 14, 20, 25,  9, 13, 30, 4,  12, 18, 21, 16,  3, 15, 22, 7])
w = torch.randint(1, T + 1, (B,))
mask = (torch.arange(T)[None, :] < w[:, None]).float()
old = torch.randn(B, T) * 0.5 - 2
logps = old + torch.randn(B, T) * 0.4
out = {fn}(logps, old, rewards, lengths, mask, G, max_len=max_len, cache_len=cache_len,
           eps_low=eps_low, eps_high=eps_high, eps=eps)

# Independent reference with explicit loops
S = max_len - cache_len
def pen(L):
    if L <= S:
        return 0.0
    if L <= max_len:
        return (S - L) / cache_len
    return -1.0
num, den, lo, hi = 0.0, 0, 0, 0
for p in range(P):
    raw = [rewards[p * G + j].item() for j in range(G)]
    if max(raw) == min(raw):
        continue
    sh = [raw[j] + pen(int(lengths[p * G + j])) for j in range(G)]
    mu = sum(sh) / G
    sd = (sum((x - mu) ** 2 for x in sh) / (G - 1)) ** 0.5
    for j in range(G):
        i = p * G + j
        a = (sh[j] - mu) / (sd + eps)
        for t in range(int(w[i])):
            rho = torch.exp(logps[i, t] - old[i, t]).item()
            c = min(max(rho, 1 - eps_low), 1 + eps_high)
            lo += int(rho < 1 - eps_low); hi += int(rho > 1 + eps_high)
            num += min(rho * a, c * a)
            den += 1
ref = -num / den
assert lo > 0 and hi > 0, 'test setup: ratios must cross both clip bounds'
assert abs(out.item() - ref) < 1e-5, f'got {out.item():.6f}, expected {ref:.6f}'
""",
        },
        {
            "name": "Gradient at ratio 1: shaped advantage over kept tokens",
            "code": """
import torch
# logps == old_logps, so dL/dlogps_{i,t} = -A_i / N_kept for kept rows and 0 for dropped rows.
# Lengths sit on every branch of the penalty, including both boundaries S and L_max.
G, T = 3, 4
max_len, cache_len = 10, 4            # S = 6
rewards = torch.tensor([1., 0., 1.,   0., 0., 0.,   1., 0., 0.])
lengths = torch.tensor([6, 8, 10,     2, 9, 12,     11, 7, 3])
B = rewards.numel()
widths = [4, 2, 3, 1, 4, 2, 3, 4, 1]
mask = torch.zeros(B, T)
for i, n in enumerate(widths):
    mask[i, :n] = 1
torch.manual_seed(2)
base = torch.randn(B, T) * 0.3 - 1
logps = base.clone().requires_grad_(True)
{fn}(logps, base.clone(), rewards, lengths, mask, G, max_len=max_len, cache_len=cache_len).backward()
pen = torch.tensor([0., -0.5, -1.,   0., -0.75, -1.,   -1., -0.25, 0.])
rt = (rewards + pen).view(-1, G)
A = ((rt - rt.mean(1, keepdim=True)) / (rt.std(1, keepdim=True) + 1e-6)).view(-1)
keep = torch.tensor([1., 1., 1., 0., 0., 0., 1., 1., 1.])
n_kept = sum(n for n, k in zip(widths, keep.tolist()) if k)
want = torch.zeros(B, T)
for i, n in enumerate(widths):
    if keep[i]:
        want[i, :n] = -A[i] / n_kept
assert logps.grad is not None, 'No gradient for logps'
assert torch.allclose(logps.grad, want, atol=1e-6), f'grad max diff {(logps.grad - want).abs().max().item():.2e}; check the penalty branches, the unbiased std, filtering on raw rewards and the token-mean over kept rows'
""",
        },
        {
            "name": "Clip-higher: only the upper bound is wider",
            "code": """
import torch
# One group, rewards [1, 0]: A = +0.7071 for row 0, -0.7071 for row 1 (unbiased std, eps tiny).
# A > 0 tokens stop getting gradient only above 1 + eps_high; A < 0 tokens only below 1 - eps_low.
G, T = 2, 4
rewards = torch.tensor([1.0, 0.0])
lengths = torch.tensor([3, 3])
mask = torch.ones(2, T)
ratios = torch.tensor([[1.10, 1.25, 1.33, 1.45],     # A > 0 row
                       [0.95, 0.85, 0.75, 0.65]])    # A < 0 row
old = torch.full((2, T), -1.0)
# (None, None) calls without clip arguments, so the signature defaults must be 0.2 / 0.28
for eps_low, eps_high in [(None, None), (0.2, 0.28), (0.3, 0.4)]:
    logps = (old + ratios.log()).requires_grad_(True)
    if eps_low is None:
        {fn}(logps, old, rewards, lengths, mask, G, max_len=10, cache_len=2).backward()
        eps_low, eps_high = 0.2, 0.28
    else:
        {fn}(logps, old, rewards, lengths, mask, G, max_len=10, cache_len=2, eps_low=eps_low, eps_high=eps_high).backward()
    g = logps.grad
    for t in range(T):
        up = ratios[0, t].item() > 1 + eps_high
        down = ratios[1, t].item() < 1 - eps_low
        assert (g[0, t] == 0) == up, f'eps_high={eps_high}: A>0 token with ratio {ratios[0, t].item()} should {"" if up else "not "}be clipped'
        assert (g[1, t] == 0) == down, f'eps_low={eps_low}: A<0 token with ratio {ratios[1, t].item()} should {"" if down else "not "}be clipped'
""",
        },
        {
            "name": "All groups filtered returns a connected zero",
            "code": """
import torch
torch.manual_seed(3)
G, T = 2, 3
rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
lengths = torch.tensor([3, 50, 2, 40])      # lengths differ, but raw rewards are uniform per group
old = torch.randn(4, T)
logps = (old + 0.1).requires_grad_(True)
out = {fn}(logps, old, rewards, lengths, torch.ones(4, T), G, max_len=10, cache_len=4)
assert isinstance(out, torch.Tensor) and out.dim() == 0, 'Must return a scalar tensor'
assert out.item() == 0.0, f'No kept group should give 0.0, got {out.item()}'
out.backward()
assert logps.grad is not None, 'The zero must stay connected to logps (grad should be zeros, not None)'
assert logps.grad.abs().max() == 0, 'Gradient should be all zeros when every group is filtered'
""",
        },
        {
            "name": "Gradient reaches only logps; padding ignored; bool mask; no mutation",
            "code": """
import torch
torch.manual_seed(4)
G, B, T = 2, 6, 5
mask = torch.ones(B, T, dtype=torch.bool)
mask[0, 2:] = False; mask[3, 1:] = False
old = torch.randn(B, T) * 0.3 - 1
lp = old + 0.2 * torch.randn(B, T)
rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
lengths = torch.tensor([4, 9, 2, 11, 7, 5])
saved = [lp.clone(), old.clone(), rewards.clone(), lengths.clone(), mask.clone()]
logps = lp.clone().requires_grad_(True)
old_g = old.clone().requires_grad_(True)
rew_g = rewards.clone().requires_grad_(True)
out1 = {fn}(logps, old_g, rew_g, lengths, mask, G, max_len=10, cache_len=4)
out1.backward()
assert logps.grad.abs().max() > 0, 'logps must receive a gradient'
for name, t in [('old_logps', old_g), ('rewards', rew_g)]:
    assert t.grad is None or t.grad.abs().max() == 0, f'{name} must be treated as a constant (detach it)'
assert logps.grad[0, 2:].abs().max() == 0 and logps.grad[3, 1:].abs().max() == 0, 'Padding must get zero gradient'
for name, before, after in zip(['logps', 'old_logps', 'rewards', 'lengths', 'mask'], saved, [lp, old, rewards, lengths, mask]):
    assert torch.equal(before, after), f'{name} was modified in place'
pad = ~mask
noisy_lp = lp.masked_fill(pad, 0.0) + pad * torch.randn(B, T)
noisy_old = old.masked_fill(pad, 0.0) + pad * torch.randn(B, T)
out2 = {fn}(noisy_lp, noisy_old, rewards, lengths, mask.float(), G, max_len=10, cache_len=4)
assert torch.allclose(out1.detach(), out2, atol=1e-6), 'Result depends on padding positions'
""",
        },
    ],
    "solution": '''def dapo_loss(logps, old_logps, rewards, lengths, mask, group_size, max_len, cache_len,
              eps_low=0.2, eps_high=0.28, eps=1e-6):
    G = group_size
    old_logps = old_logps.detach()
    r = rewards.detach().to(logps.dtype)

    # 1. Dynamic sampling: a group whose raw rewards are all equal has zero advantage
    raw = r.view(-1, G)
    keep = raw.max(dim=1).values > raw.min(dim=1).values          # (P,)
    if not keep.any():
        return (logps * 0).sum()                                    # zero, still connected to logps

    # 2. Overlong shaping: 0 up to S, linear down to -1 at max_len, -1 beyond
    L = lengths.to(logps.dtype)
    soft = max_len - cache_len
    penalty = torch.where(L <= soft, torch.zeros_like(L),
                          torch.where(L <= max_len, (soft - L) / cache_len, -torch.ones_like(L)))
    shaped = (r + penalty).view(-1, G)

    # 3. Group-relative advantage on the shaped rewards (dropped groups are masked out below)
    adv = (shaped - shaped.mean(dim=1, keepdim=True)) / (shaped.std(dim=1, keepdim=True) + eps)
    adv = adv.view(-1, 1)

    # 4. Clip-higher: the upper bound 1 + eps_high is wider than the lower 1 - eps_low
    ratio = torch.exp(logps - old_logps)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv)

    # 5. Token-mean over every valid token of the kept rows
    m = mask.to(logps.dtype) * keep.repeat_interleave(G).to(logps.dtype)[:, None]
    return -(surr * m).sum() / m.sum()''',
    "demo": """import torch

torch.manual_seed(0)
G, P, T = 4, 3, 8
B = G * P
rewards = torch.tensor([1., 0., 1., 0.,   1., 1., 1., 1.,   0., 0., 1., 0.])
lengths = torch.tensor([300, 900, 1500, 2100,  400, 500, 600, 700,  800, 1900, 1200, 2500])
mask = torch.ones(B, T)
old = torch.randn(B, T) * 0.3 - 1
logps = (old + 0.1 * torch.randn(B, T)).requires_grad_(True)

loss = dapo_loss(logps, old, rewards, lengths, mask, G, max_len=2048, cache_len=512)
loss.backward()
print('loss:', round(loss.item(), 6))
print('rows with gradient:', (logps.grad.abs().sum(1) > 0).int().tolist())
print('(group 2 is all correct and is dropped by dynamic sampling)')""",

}
