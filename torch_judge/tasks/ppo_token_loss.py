"""Token-Level PPO Actor-Critic Loss task."""

TASK = {
    "title": "Token-Level PPO Actor-Critic Loss",
    "title_zh": "完整版 PPO 损失",
    "difficulty": "Hard",
    "description_en": "Implement the full token-level PPO loss used in RLHF: the clipped policy loss, the clipped value loss, and the two statistics every training dashboard shows.\n\nIn a language model every generated token is an action, so ratios, advantages and values are all per token, and padding must be masked out. verl and OpenRLHF compute exactly these two terms, on separate actor and critic models with separate optimizers; adding them as `pg + vf_coef * vf` is the shared-backbone form of the original PPO paper and TRL / trlx. The advantages and returns come from GAE.\n\n**Signature:** `ppo_token_loss(logps, old_logps, advantages, values, old_values, returns, mask, clip_ratio=0.2, value_clip=0.2, vf_coef=0.5) -> dict[str, Tensor]`\n\n**Parameters:**\n- `logps` — current policy log-probs of the sampled tokens (B, T)\n- `old_logps` — log-probs under the policy that generated the batch (B, T), constant\n- `advantages` — per-token advantages (B, T), constant\n- `values` — current critic values (B, T)\n- `old_values` — critic values at rollout time (B, T), constant\n- `returns` — value targets (B, T), constant\n- `mask` — (B, T), 1 on response tokens, 0 on padding (float or bool)\n- `clip_ratio` — $\\epsilon$ for the policy ratio; `value_clip` — clip range for the value; `vf_coef` — value-loss weight\n\n**Returns:** a dict of scalar tensors with keys `\"loss\"`, `\"pg_loss\"`, `\"vf_loss\"`, `\"clipfrac\"`, `\"approx_kl\"`\n\n**Formulas** (\"masked token-mean\" $= \\sum m \\cdot x / \\sum m$ over the whole batch):\n$$\n\\begin{aligned}\n\\rho_t &= \\exp(\\text{logps}_t - \\text{old\\_logps}_t) \\\\\n\\text{pg}_t &= \\max\\bigl(-A_t \\rho_t,\\ -A_t \\operatorname{clip}(\\rho_t, 1-\\epsilon, 1+\\epsilon)\\bigr) \\\\\nv^{\\text{clip}}_t &= \\text{old\\_values}_t + \\operatorname{clip}(\\text{values}_t - \\text{old\\_values}_t,\\ -c_v,\\ c_v) \\\\\n\\text{vf}_t &= \\tfrac{1}{2}\\max\\bigl((\\text{values}_t - R_t)^2,\\ (v^{\\text{clip}}_t - R_t)^2\\bigr)\n\\end{aligned}\n$$\n- `pg_loss`, `vf_loss` — masked token-means of $\\text{pg}_t$ and $\\text{vf}_t$\n- `loss` $=$ `pg_loss` $+$ `vf_coef` $\\cdot$ `vf_loss`\n- `clipfrac` — masked token-mean of $\\mathbb{1}\\bigl[-A_t \\operatorname{clip}(\\rho_t) > -A_t \\rho_t\\bigr]$, the fraction of tokens where the clipped branch is active\n- `approx_kl` — masked token-mean of $\\text{old\\_logps}_t - \\text{logps}_t$\n\n**Constraints:**\n- No dual-clip: the policy loss for $A < 0$ is not capped (verl applies dual-clip with $c = 3$ by default).\n- Gradients flow only into `logps` and `values`; every other input is a constant.\n- Padding positions (mask 0) hold arbitrary finite values and must not affect any output.\n- Every batch has at least one response token. Do not modify the inputs.",
    "description_zh": "实现 RLHF 中完整的逐 token PPO 损失：裁剪后的策略损失、裁剪后的价值损失，以及训练监控里必看的两个统计量。\n\n语言模型里每个生成的 token 都是一个动作，所以比率、优势和价值都按 token 计算，padding 必须被掩掉。verl 和 OpenRLHF 计算的正是这两项，只是 actor 和 critic 是两个独立的模型、各自用自己的优化器；把它们加成 `pg + vf_coef * vf` 是共享主干时的写法，见原始 PPO 论文和 TRL / trlx。优势和回报来自 GAE。\n\n**签名:** `ppo_token_loss(logps, old_logps, advantages, values, old_values, returns, mask, clip_ratio=0.2, value_clip=0.2, vf_coef=0.5) -> dict[str, Tensor]`\n\n**参数:**\n- `logps` — 当前策略对采样 token 的对数概率 (B, T)\n- `old_logps` — 生成这批数据的策略给出的对数概率 (B, T)，常数\n- `advantages` — 逐 token 优势 (B, T)，常数\n- `values` — 当前 critic 的价值 (B, T)\n- `old_values` — 采样时 critic 的价值 (B, T)，常数\n- `returns` — 价值的回归目标 (B, T)，常数\n- `mask` — (B, T)，回答 token 为 1，padding 为 0（float 或 bool）\n- `clip_ratio` — 策略比率的 $\\epsilon$；`value_clip` — 价值的裁剪范围；`vf_coef` — 价值损失的权重\n\n**返回:** 标量张量组成的字典，键为 `\"loss\"`、`\"pg_loss\"`、`\"vf_loss\"`、`\"clipfrac\"`、`\"approx_kl\"`\n\n**公式**（「掩码 token 平均」$= \\sum m \\cdot x / \\sum m$，对整个 batch 求）：\n$$\n\\begin{aligned}\n\\rho_t &= \\exp(\\text{logps}_t - \\text{old\\_logps}_t) \\\\\n\\text{pg}_t &= \\max\\bigl(-A_t \\rho_t,\\ -A_t \\operatorname{clip}(\\rho_t, 1-\\epsilon, 1+\\epsilon)\\bigr) \\\\\nv^{\\text{clip}}_t &= \\text{old\\_values}_t + \\operatorname{clip}(\\text{values}_t - \\text{old\\_values}_t,\\ -c_v,\\ c_v) \\\\\n\\text{vf}_t &= \\tfrac{1}{2}\\max\\bigl((\\text{values}_t - R_t)^2,\\ (v^{\\text{clip}}_t - R_t)^2\\bigr)\n\\end{aligned}\n$$\n- `pg_loss`、`vf_loss` — $\\text{pg}_t$ 与 $\\text{vf}_t$ 的掩码 token 平均\n- `loss` $=$ `pg_loss` $+$ `vf_coef` $\\cdot$ `vf_loss`\n- `clipfrac` — $\\mathbb{1}\\bigl[-A_t \\operatorname{clip}(\\rho_t) > -A_t \\rho_t\\bigr]$ 的掩码 token 平均，即裁剪分支生效的 token 比例\n- `approx_kl` — $\\text{old\\_logps}_t - \\text{logps}_t$ 的掩码 token 平均\n\n**约束:**\n- 不做 dual-clip：$A < 0$ 时策略损失不设上限（verl 默认会用 $c = 3$ 的 dual-clip）。\n- 梯度只流向 `logps` 和 `values`，其余输入都是常数。\n- padding 位置（mask 为 0）可以是任意有限值，不得影响任何输出。\n- 每个 batch 至少有一个回答 token。不要修改输入。",
    "function_name": "ppo_token_loss",
    "hint": "m = mask.to(logps.dtype); mean = lambda x: (x * m).sum() / m.sum()\nratio = exp(logps - old_logps.detach())\npg1 = -A * ratio;  pg2 = -A * clamp(ratio, 1-eps, 1+eps)\npg_loss = mean(max(pg1, pg2))\nv_clip = old_values + clamp(values - old_values, -c, c)\nvf_loss = mean(0.5 * max((values-R)**2, (v_clip-R)**2))\nclipfrac = mean(pg2 > pg1); approx_kl = mean(old_logps - logps)",
    "hint_zh": "m = mask.to(logps.dtype); mean = lambda x: (x * m).sum() / m.sum()\nratio = exp(logps - old_logps.detach())\npg1 = -A * ratio;  pg2 = -A * clamp(ratio, 1-eps, 1+eps)\npg_loss = mean(max(pg1, pg2))\nv_clip = old_values + clamp(values - old_values, -c, c)\nvf_loss = mean(0.5 * max((values-R)**2, (v_clip-R)**2))\nclipfrac = mean(pg2 > pg1); approx_kl = mean(old_logps - logps)",
    "tests": [
        {
            "name": "Returns a dict of scalar tensors",
            "code": """
import torch
torch.manual_seed(0)
B, T = 3, 5
logps = (torch.randn(B, T) * 0.1 - 1).requires_grad_(True)
values = torch.randn(B, T).requires_grad_(True)
out = {fn}(logps, logps.detach() + 0.05 * torch.randn(B, T), torch.randn(B, T), values,
           values.detach() + 0.1 * torch.randn(B, T), torch.randn(B, T), torch.ones(B, T))
assert isinstance(out, dict), f'Expected a dict, got {type(out).__name__}'
for key in ['loss', 'pg_loss', 'vf_loss', 'clipfrac', 'approx_kl']:
    assert key in out, f'Missing key {key!r}'
    assert isinstance(out[key], torch.Tensor) and out[key].dim() == 0, f'{key} must be a scalar tensor'
    assert torch.isfinite(out[key]), f'{key} is not finite'
assert out['loss'].requires_grad, 'loss must carry a gradient'
""",
        },
        {
            "name": "All outputs match a reference with padding",
            "code": """
import torch
torch.manual_seed(1)
B, T, eps, cv, c1 = 4, 7, 0.3, 0.1, 0.7
lengths = [7, 3, 5, 1]
mask = torch.zeros(B, T)
for b, n in enumerate(lengths):
    mask[b, :n] = 1
old_logps = torch.randn(B, T) * 0.3 - 1.5
logps = old_logps + torch.randn(B, T) * 0.4
adv = torch.randn(B, T)
old_values = torch.randn(B, T)
values = old_values + torch.randn(B, T) * 0.5
returns = torch.randn(B, T)
out = {fn}(logps, old_logps, adv, values, old_values, returns, mask, clip_ratio=eps, value_clip=cv, vf_coef=c1)

# Independent reference: loop over valid tokens only
pg, vf, clipped, kl = [], [], [], []
for b in range(B):
    for t in range(lengths[b]):
        r = torch.exp(logps[b, t] - old_logps[b, t])
        a = adv[b, t]
        u, c = -a * r, -a * r.clamp(1 - eps, 1 + eps)
        pg.append(max(u, c)); clipped.append(float(c > u))
        vc = old_values[b, t] + (values[b, t] - old_values[b, t]).clamp(-cv, cv)
        vf.append(0.5 * max((values[b, t] - returns[b, t]) ** 2, (vc - returns[b, t]) ** 2))
        kl.append(old_logps[b, t] - logps[b, t])
pg_ref, vf_ref = torch.stack(pg).mean(), torch.stack(vf).mean()
ref = {'pg_loss': pg_ref, 'vf_loss': vf_ref, 'loss': pg_ref + c1 * vf_ref,
       'clipfrac': torch.tensor(sum(clipped) / len(clipped)), 'approx_kl': torch.stack(kl).mean()}
for key, want in ref.items():
    got = out[key].detach()
    assert torch.allclose(got, want, atol=1e-5), f'{key}: got {got.item():.6f}, expected {want.item():.6f}'
""",
        },
        {
            "name": "Value loss takes the pessimistic (max) branch",
            "code": """
import torch
# values moved far from old_values toward the target: the clipped value is still far from the target,
# so the max must pick the clipped error
old_values = torch.tensor([[0.0, 0.0]])
values = torch.tensor([[0.9, -0.05]])
returns = torch.tensor([[1.0, 0.0]])
z = torch.zeros(1, 2)
out = {fn}(z.clone(), z.clone(), z.clone(), values, old_values, returns, torch.ones(1, 2), value_clip=0.2)
# token 0: unclipped (0.9-1)^2=0.01, clipped (0.2-1)^2=0.64 -> 0.64; token 1: both 0.0025
want = 0.5 * (0.64 + 0.0025) / 2
assert abs(out['vf_loss'].item() - want) < 1e-6, f'vf_loss: got {out[\"vf_loss\"].item():.6f}, expected {want:.6f} (max of clipped/unclipped)'
""",
        },
        {
            "name": "clipfrac counts only tokens where clipping is active",
            "code": """
import torch
eps = 0.2
# ratio = e^0.5 ~ 1.65 > 1 + eps on every token
old_logps = torch.zeros(1, 4)
logps = torch.full((1, 4), 0.5)
# A > 0: the clipped branch wins -> counted.  A < 0: the unclipped branch is larger -> not counted
adv = torch.tensor([[1.0, 2.0, -1.0, -2.0]])
z = torch.zeros(1, 4)
out = {fn}(logps, old_logps, adv, z.clone(), z.clone(), z.clone(), torch.ones(1, 4), clip_ratio=eps)
assert abs(out['clipfrac'].item() - 0.5) < 1e-6, f'clipfrac: got {out[\"clipfrac\"].item():.4f}, expected 0.5 (ratio out of range with A<0 is not clipped)'
# ratio below 1 - eps: now A < 0 is the clipped side
out2 = {fn}(-logps, old_logps, adv, z.clone(), z.clone(), z.clone(), torch.ones(1, 4), clip_ratio=eps)
assert abs(out2['clipfrac'].item() - 0.5) < 1e-6, f'clipfrac with ratio < 1-eps: got {out2[\"clipfrac\"].item():.4f}, expected 0.5'
# Inside the range nothing is clipped
out3 = {fn}(torch.full((1, 4), 0.05), old_logps, adv, z.clone(), z.clone(), z.clone(), torch.ones(1, 4), clip_ratio=eps)
assert out3['clipfrac'].item() == 0.0, f'clipfrac inside the range should be 0, got {out3[\"clipfrac\"].item():.4f}'
""",
        },
        {
            "name": "Gradients reach only logps and values",
            "code": """
import torch
torch.manual_seed(2)
B, T = 3, 6
mask = torch.ones(B, T); mask[1, 4:] = 0; mask[2, 2:] = 0
base_old = torch.randn(B, T) * 0.3
logps = (base_old + torch.randn(B, T) * 0.3).requires_grad_(True)
values = torch.randn(B, T).requires_grad_(True)
old_logps = base_old.clone().requires_grad_(True)
adv = torch.randn(B, T).requires_grad_(True)
old_values = (values.detach() + torch.randn(B, T) * 0.3).requires_grad_(True)
returns = torch.randn(B, T).requires_grad_(True)
eps, cv, c1 = 0.3, 0.1, 0.7
out = {fn}(logps, old_logps, adv, values, old_values, returns, mask, clip_ratio=eps, value_clip=cv, vf_coef=c1)
out['loss'].backward()
assert logps.grad is not None and values.grad is not None, 'logps and values must receive gradients'
for name, t in [('old_logps', old_logps), ('advantages', adv), ('old_values', old_values), ('returns', returns)]:
    assert t.grad is None or t.grad.abs().max() == 0, f'{name} must be treated as a constant (detach it)'
# Compare against autograd through an independent reference
lp = logps.detach().clone().requires_grad_(True)
v = values.detach().clone().requires_grad_(True)
o, a, ov, R = old_logps.detach(), adv.detach(), old_values.detach(), returns.detach()
r = torch.exp(lp - o)
pg = torch.where(mask.bool(), torch.maximum(-a * r, -a * r.clamp(1 - eps, 1 + eps)), torch.zeros(()))
vc = ov + (v - ov).clamp(-cv, cv)
vf = torch.where(mask.bool(), 0.5 * torch.maximum((v - R) ** 2, (vc - R) ** 2), torch.zeros(()))
(pg.sum() / mask.sum() + c1 * vf.sum() / mask.sum()).backward()
assert torch.allclose(logps.grad, lp.grad, atol=1e-6), f'logps grad max diff {(logps.grad - lp.grad).abs().max().item():.2e}'
assert torch.allclose(values.grad, v.grad, atol=1e-6), f'values grad max diff {(values.grad - v.grad).abs().max().item():.2e}'
""",
        },
        {
            "name": "Padding values do not change any output; bool mask works",
            "code": """
import torch
torch.manual_seed(3)
B, T = 2, 5
mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
args = [torch.randn(B, T) * 0.3 for _ in range(6)]
out1 = {fn}(*args, mask)
pad = ~mask
noisy = [a.masked_fill(pad, 0.0) + pad * torch.randn(B, T) * 0.7 for a in args]
out2 = {fn}(*noisy, mask.float())
for key in ['loss', 'pg_loss', 'vf_loss', 'clipfrac', 'approx_kl']:
    assert torch.allclose(out1[key].detach(), out2[key].detach(), atol=1e-6), f'{key} depends on padding positions'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(4)
B, T = 2, 4
args = [torch.randn(B, T) * 0.3 for _ in range(6)]
mask = torch.ones(B, T); mask[0, 3] = 0
saved = [a.clone() for a in args] + [mask.clone()]
{fn}(*args, mask)
for name, before, after in zip(['logps', 'old_logps', 'advantages', 'values', 'old_values', 'returns', 'mask'], saved, args + [mask]):
    assert torch.equal(before, after), f'{name} was modified in place'
""",
        },
    ],
    "solution": '''def ppo_token_loss(logps, old_logps, advantages, values, old_values, returns, mask,
                   clip_ratio=0.2, value_clip=0.2, vf_coef=0.5):
    m = mask.to(logps.dtype)
    n_tokens = m.sum()

    def masked_mean(x):
        # Token-mean over every response token in the batch
        return (x * m).sum() / n_tokens

    old_logps, advantages = old_logps.detach(), advantages.detach()
    old_values, returns = old_values.detach(), returns.detach()

    # Policy loss: negate the PPO objective, so min becomes max
    ratio = torch.exp(logps - old_logps)
    pg_unclipped = -advantages * ratio
    pg_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_loss = masked_mean(torch.maximum(pg_unclipped, pg_clipped))

    # Value loss: the new value may move at most value_clip away from the rollout value
    v_clipped = old_values + torch.clamp(values - old_values, -value_clip, value_clip)
    vf = 0.5 * torch.maximum((values - returns) ** 2, (v_clipped - returns) ** 2)
    vf_loss = masked_mean(vf)

    with torch.no_grad():
        clipfrac = masked_mean((pg_clipped > pg_unclipped).to(logps.dtype))
        approx_kl = masked_mean(old_logps - logps)

    return {
        "loss": pg_loss + vf_coef * vf_loss,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "clipfrac": clipfrac,
        "approx_kl": approx_kl,
    }''',
    "demo": """import torch

torch.manual_seed(0)
B, T = 4, 6
mask = torch.ones(B, T)
mask[1, 4:] = 0
mask[3, 2:] = 0
old_logps = torch.randn(B, T) * 0.3 - 1.0
logps = (old_logps + torch.randn(B, T) * 0.3).requires_grad_(True)
advantages = torch.randn(B, T)
old_values = torch.randn(B, T)
values = (old_values + torch.randn(B, T) * 0.3).requires_grad_(True)
returns = old_values + advantages

out = ppo_token_loss(logps, old_logps, advantages, values, old_values, returns, mask)
for key, val in out.items():
    print(f'{key:>9}: {val.item():.4f}')
out['loss'].backward()
print('logps grad on padding:', logps.grad[1, 4:].tolist())""",

}
