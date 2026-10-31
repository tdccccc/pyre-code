"""GSPO (Group Sequence Policy Optimization) loss task."""

TASK = {
    "title": "GSPO Sequence-Level Ratio Loss",
    "title_zh": "GSPO 序列级比率损失",
    "difficulty": "Medium",
    "description_en": "Implement the GSPO loss (Group Sequence Policy Optimization, Qwen team 2025, used to train Qwen3). GRPO computes an importance ratio for every token, although the reward and the advantage belong to the whole response. GSPO uses one ratio per response instead: the length-normalized sequence likelihood ratio, i.e. the geometric mean of the token ratios. Clipping then keeps or drops whole responses.\n\n**Signature:** `gspo_loss(logps, old_logps, advantages, mask, clip_low=3e-4, clip_high=4e-4) -> Tensor`\n\n**Parameters:**\n- `logps` — current policy log-probs of the sampled tokens (B, T)\n- `old_logps` — log-probs under the policy that generated the batch (B, T), constant\n- `advantages` — one advantage per response (B,), constant (e.g. group-normalized rewards)\n- `mask` — (B, T), 1 on response tokens, 0 on padding (float or bool); every row has at least one token\n- `clip_low`, `clip_high` — the clip range is $[1 - \\texttt{clip\\_low},\\ 1 + \\texttt{clip\\_high}]$\n\n**Returns:** scalar loss\n\n**Formulas:** with $|o_i| = \\sum_t m_{i,t}$\n$$\n\\begin{aligned}\ns_i &= \\exp\\Bigl(\\frac{1}{|o_i|}\\sum_t m_{i,t}\\,(\\text{logps}_{i,t} - \\text{old\\_logps}_{i,t})\\Bigr) \\\\\nL &= -\\frac{1}{B}\\sum_i \\min\\bigl(s_i A_i,\\ \\operatorname{clip}(s_i,\\ 1 - \\texttt{clip\\_low},\\ 1 + \\texttt{clip\\_high})\\, A_i\\bigr)\n\\end{aligned}\n$$\n\n**Constraints:**\n- Gradients flow only into `logps`.\n- Padding positions hold arbitrary finite values and must not affect the result. Do not modify the inputs.",
    "description_zh": "实现 GSPO 损失（Group Sequence Policy Optimization，Qwen 团队 2025 年提出，用于训练 Qwen3）。GRPO 给每个 token 算一个重要性比率，可奖励和优势都属于整条回答。GSPO 改为每个回答只用一个比率：按长度归一化的序列似然比，也就是各 token 比率的几何平均。裁剪也因此以整条回答为单位，要么整条保留，要么整条丢弃。\n\n**签名:** `gspo_loss(logps, old_logps, advantages, mask, clip_low=3e-4, clip_high=4e-4) -> Tensor`\n\n**参数:**\n- `logps` — 当前策略对采样 token 的对数概率 (B, T)\n- `old_logps` — 生成这批数据的策略给出的对数概率 (B, T)，常数\n- `advantages` — 每个回答一个优势 (B,)，常数（例如组内归一化的奖励）\n- `mask` — (B, T)，回答 token 为 1，padding 为 0（float 或 bool）；每行至少有一个 token\n- `clip_low`、`clip_high` — 裁剪区间为 $[1 - \\texttt{clip\\_low},\\ 1 + \\texttt{clip\\_high}]$\n\n**返回:** 标量损失\n\n**公式:** 记 $|o_i| = \\sum_t m_{i,t}$\n$$\n\\begin{aligned}\ns_i &= \\exp\\Bigl(\\frac{1}{|o_i|}\\sum_t m_{i,t}\\,(\\text{logps}_{i,t} - \\text{old\\_logps}_{i,t})\\Bigr) \\\\\nL &= -\\frac{1}{B}\\sum_i \\min\\bigl(s_i A_i,\\ \\operatorname{clip}(s_i,\\ 1 - \\texttt{clip\\_low},\\ 1 + \\texttt{clip\\_high})\\, A_i\\bigr)\n\\end{aligned}\n$$\n\n**约束:**\n- 梯度只流向 `logps`。\n- padding 位置可以是任意有限值，不得影响结果。不要修改输入。",
    "function_name": "gspo_loss",
    "hint": "m = mask.float()\nlog_s = ((logps - old_logps) * m).sum(1) / m.sum(1)   # (B,)\ns = exp(log_s)\nA = advantages.detach()\nsurr = min(s * A, clamp(s, 1 - clip_low, 1 + clip_high) * A)\nloss = -surr.mean()",
    "hint_zh": "m = mask.float()\nlog_s = ((logps - old_logps) * m).sum(1) / m.sum(1)   # (B,)\ns = exp(log_s)\nA = advantages.detach()\nsurr = min(s * A, clamp(s, 1 - clip_low, 1 + clip_high) * A)\nloss = -surr.mean()",
    "tests": [
        {
            "name": "Returns a finite scalar with a gradient",
            "code": """
import torch
torch.manual_seed(0)
B, T = 4, 5
old = torch.randn(B, T) * 0.3 - 1
logps = (old + 1e-5 * torch.randn(B, T)).requires_grad_(True)
out = {fn}(logps, old, torch.randn(B), torch.ones(B, T))
assert isinstance(out, torch.Tensor) and out.dim() == 0, 'Loss must be a scalar tensor'
assert torch.isfinite(out), 'Loss should be finite'
assert out.requires_grad, 'Loss must carry a gradient'
""",
        },
        {
            "name": "Matches a reference with padding and clipping on both sides",
            "code": """
import torch
torch.manual_seed(1)
B, T = 16, 7
clip_low, clip_high = 0.05, 0.1
lengths = torch.randint(1, T + 1, (B,))
mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
old = torch.randn(B, T) * 0.5 - 2
logps = old + torch.randn(B, T) * 0.15
adv = torch.randn(B)
out = {fn}(logps, old, adv, mask, clip_low=clip_low, clip_high=clip_high)

# Independent reference: explicit loop over responses and their real tokens
terms, hit_low, hit_high = [], 0, 0
for i in range(B):
    n = int(lengths[i])
    log_s = sum(float(logps[i, t] - old[i, t]) for t in range(n)) / n
    s = torch.exp(torch.tensor(log_s))
    a = adv[i]
    unclipped = s * a
    clipped = s.clamp(1 - clip_low, 1 + clip_high) * a
    hit_low += int(s < 1 - clip_low)
    hit_high += int(s > 1 + clip_high)
    terms.append(torch.minimum(unclipped, clipped))
ref = -torch.stack(terms).mean()
assert hit_low > 0 and hit_high > 0, 'test setup: sequences must leave the clip range on both sides'
assert torch.allclose(out, ref, atol=1e-5), f'got {out.item():.6f}, expected {ref.item():.6f}'
""",
        },
        {
            "name": "Gradient inside the clip range: -A_i * s_i / (B * |o_i|) per token",
            "code": """
import torch
# A wide clip range so no response is clipped; d s_i / d logps_{i,t} = s_i / |o_i|
torch.manual_seed(2)
B, T = 5, 6
lengths = [6, 1, 3, 4, 2]
mask = torch.zeros(B, T)
for i, n in enumerate(lengths):
    mask[i, :n] = 1
old = torch.randn(B, T) * 0.4 - 1
base = old + torch.randn(B, T) * 0.3
adv = torch.tensor([1.0, -0.5, 2.0, -1.5, 0.7])
logps = base.clone().requires_grad_(True)
{fn}(logps, old, adv, mask, clip_low=0.95, clip_high=10.0).backward()
want = torch.zeros(B, T)
for i, n in enumerate(lengths):
    s = torch.exp((base[i, :n] - old[i, :n]).mean())
    want[i, :n] = -adv[i] * s / (B * n)
assert logps.grad is not None, 'No gradient for logps'
assert torch.allclose(logps.grad, want, atol=1e-6), f'grad max diff {(logps.grad - want).abs().max().item():.2e}; s_i must be exp(mean of the token log-ratios), the geometric mean'
""",
        },
        {
            "name": "Length-normalized: same per-token log-ratio gives the same s",
            "code": """
import torch
# Every token has log-ratio 0.02, so s_i = exp(0.02) for every response, whatever its length.
# Summing the log-ratios instead would give exp(0.02 * |o_i|).
B, T, d = 3, 8, 0.02
lengths = [1, 4, 8]
mask = torch.zeros(B, T)
for i, n in enumerate(lengths):
    mask[i, :n] = 1
old = torch.full((B, T), -1.5)
logps = old + d
adv = torch.tensor([1.0, 1.0, 1.0])
out = {fn}(logps, old, adv, mask, clip_low=0.2, clip_high=0.2)
want = -torch.exp(torch.tensor(d))
assert torch.allclose(out, want, atol=1e-6), f'got {out.item():.6f}, expected {want.item():.6f}: s_i must not grow with the response length'
""",
        },
        {
            "name": "Whole responses are clipped (zero grad), pessimistic side kept",
            "code": """
import torch
B, T = 4, 3
clip_low, clip_high = 0.1, 0.2
old = torch.zeros(B, T) - 1.0
# Per-row constant log-ratio -> s = exp(delta)
delta = torch.tensor([0.4, -0.4, -0.4, 0.4])   # s ~ 1.49, 0.67, 0.67, 1.49
adv = torch.tensor([1.0, -1.0, 1.0, -1.0])
logps = (old + delta[:, None]).requires_grad_(True)
{fn}(logps, old, adv, torch.ones(B, T), clip_low=clip_low, clip_high=clip_high).backward()
g = logps.grad
assert g[0].abs().max() == 0, 'A > 0 and s > 1 + clip_high: the whole response is clipped, no gradient'
assert g[1].abs().max() == 0, 'A < 0 and s < 1 - clip_low: the whole response is clipped, no gradient'
assert g[2].abs().min() > 0, 'A > 0 and s < 1 - clip_low: min keeps the unclipped term, gradient must flow'
assert g[3].abs().min() > 0, 'A < 0 and s > 1 + clip_high: min keeps the unclipped term, gradient must flow'
""",
        },
        {
            "name": "Default clip range is [1 - 3e-4, 1 + 4e-4]",
            "code": """
import torch
# No clip arguments: the defaults must be clip_low=3e-4 and clip_high=4e-4.
# Each row has a constant per-token log-ratio d, so s = exp(d).
#   d = +1e-3,   A > 0: s - 1 = 1.0e-3 > 4e-4   -> clipped (upper side)
#   d = -1e-3,   A < 0: 1 - s = 1.0e-3 > 3e-4   -> clipped (lower side)
#   d = +1e-4,   A > 0: inside                  -> gradient
#   d = -1e-4,   A < 0: inside                  -> gradient
#   d = +3.5e-4, A > 0: 3.5e-4 < clip_high=4e-4 -> gradient (clipped if clip_high were 3e-4)
#   d = -3.5e-4, A < 0: 3.5e-4 > clip_low=3e-4  -> clipped  (kept if clip_low were 4e-4)
B, T = 6, 4
old = torch.full((B, T), -1.0, dtype=torch.float64)
d = torch.tensor([1e-3, -1e-3, 1e-4, -1e-4, 3.5e-4, -3.5e-4], dtype=torch.float64)
adv = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
logps = (old + d[:, None]).requires_grad_(True)
{fn}(logps, old, adv, torch.ones(B, T, dtype=torch.float64)).backward()
g = logps.grad.abs().amax(dim=1)
assert g[0] == 0, 'd=+1e-3 with A>0 must be clipped by the default clip_high=4e-4'
assert g[1] == 0, 'd=-1e-3 with A<0 must be clipped by the default clip_low=3e-4'
assert g[2] > 0 and g[3] > 0, 'Ratios within 1e-4 of 1 must not be clipped with the default range'
assert g[4] > 0, 's = 1 + 3.5e-4 is inside the default upper bound 1 + 4e-4 (clip_high default must be 4e-4)'
assert g[5] == 0, 's = 1 - 3.5e-4 is below the default lower bound 1 - 3e-4 (clip_low default must be 3e-4)'
""",
        },
        {
            "name": "Gradient reaches only logps; padding gets none",
            "code": """
import torch
torch.manual_seed(4)
B, T = 4, 5
logps = (torch.randn(B, T) * 0.3 - 1).requires_grad_(True)
old = (logps.detach() + 1e-4 * torch.randn(B, T)).requires_grad_(True)
adv = torch.randn(B).requires_grad_(True)
mask = torch.ones(B, T); mask[1, 3:] = 0; mask[2, 1:] = 0
{fn}(logps, old, adv, mask, clip_low=0.2, clip_high=0.3).backward()
assert logps.grad is not None and logps.grad.abs().max() > 0, 'logps must receive a gradient'
for name, t in [('old_logps', old), ('advantages', adv)]:
    assert t.grad is None or t.grad.abs().max() == 0, f'{name} must be treated as a constant (detach it)'
assert logps.grad[1, 3:].abs().max() == 0 and logps.grad[2, 1:].abs().max() == 0, 'Padding positions must get zero gradient'
""",
        },
        {
            "name": "Padding is ignored; bool mask works; inputs not mutated",
            "code": """
import torch
torch.manual_seed(5)
B, T = 6, 5
mask = torch.ones(B, T, dtype=torch.bool)
mask[0, 2:] = False; mask[4, 1:] = False
old = torch.randn(B, T) * 0.3 - 1
logps = old + torch.randn(B, T) * 0.05
adv = torch.randn(B)
saved = [logps.clone(), old.clone(), adv.clone(), mask.clone()]
out1 = {fn}(logps, old, adv, mask, clip_low=0.02, clip_high=0.03)
for name, before, after in zip(['logps', 'old_logps', 'advantages', 'mask'], saved, [logps, old, adv, mask]):
    assert torch.equal(before, after), f'{name} was modified in place'
pad = ~mask
noisy_logps = logps.masked_fill(pad, 0.0) + pad * torch.randn(B, T)
noisy_old = old.masked_fill(pad, 0.0) + pad * torch.randn(B, T)
out2 = {fn}(noisy_logps, noisy_old, adv, mask.float(), clip_low=0.02, clip_high=0.03)
assert torch.allclose(out1, out2, atol=1e-6), 'Result depends on padding positions'
""",
        },
    ],
    "solution": '''def gspo_loss(logps, old_logps, advantages, mask, clip_low=3e-4, clip_high=4e-4):
    m = mask.to(logps.dtype)
    adv = advantages.detach().to(logps.dtype)

    # One ratio per response: exp of the mean token log-ratio (geometric mean of token ratios)
    log_ratio = (logps - old_logps.detach()) * m
    seq_ratio = torch.exp(log_ratio.sum(dim=1) / m.sum(dim=1))        # (B,)

    # PPO-style pessimistic clipping, but on whole responses
    clipped = torch.clamp(seq_ratio, 1 - clip_low, 1 + clip_high)
    surr = torch.minimum(seq_ratio * adv, clipped * adv)
    return -surr.mean()''',
    "demo": """import torch

torch.manual_seed(0)
B, T = 4, 6
lengths = torch.tensor([6, 2, 4, 5])
mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
old_logps = torch.randn(B, T) * 0.3 - 1.0
# Rows 0 and 1 drift past the clip range in the direction their advantage pushes; rows 2 and 3 stay inside
drift = torch.tensor([6e-4, -6e-4, 1e-4, -1e-4])[:, None]
logps = (old_logps + drift + torch.randn(B, T) * 2e-5).requires_grad_(True)
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])

loss = gspo_loss(logps, old_logps, advantages, mask)
loss.backward()
m = mask
s = torch.exp(((logps.detach() - old_logps) * m).sum(1) / m.sum(1))
print('loss:', round(loss.item(), 6))
print('sequence ratios s_i:', [round(v, 6) for v in s.tolist()])
print('clip range: [1 - 3e-4, 1 + 4e-4]; whole response clipped (zero grad):',
      [bool(logps.grad[i].abs().max() == 0) for i in range(B)])""",

}
