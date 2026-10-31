"""Full token-level GRPO loss task."""

TASK = {
    "title": "Full GRPO Loss (Clip + KL, Token-Level)",
    "title_zh": "完整版 GRPO 损失",
    "difficulty": "Hard",
    "description_en": "Implement the GRPO loss as written in DeepSeekMath: group-normalized advantages, the PPO clipped ratio per token, a k3 KL penalty to the reference model inside the loss, and a per-response token mean. In the frameworks this is TRL `loss_type=\"grpo\"` and verl `loss_agg_mode=\"seq-mean-token-mean\"`; both now default to other aggregations (token-mean variants) and TRL to `beta=0`.\n\nEach prompt is answered $G$ times. A response's advantage is its reward standardized within its own group, shared by all of its tokens, so no critic is needed.\n\n**Signature:** `grpo_token_loss(logps, old_logps, ref_logps, rewards, mask, group_size, clip_ratio=0.2, kl_coef=0.04, eps=1e-6) -> Tensor`\n\n**Parameters:**\n- `logps` — current policy log-probs of the sampled tokens (B, T)\n- `old_logps` — log-probs under the policy that generated the batch (B, T), constant\n- `ref_logps` — log-probs under the frozen reference model (B, T), constant\n- `rewards` — one scalar reward per response (B,), constant\n- `mask` — (B, T), 1 on response tokens, 0 on padding (float or bool); every row has at least one token\n- `group_size` — $G \\ge 2$; $B$ is a multiple of $G$ and the $G$ responses to one prompt are consecutive rows\n- `clip_ratio` — $\\epsilon$; `kl_coef` — $\\beta$; `eps` — added to the std\n\n**Returns:** scalar loss\n\n**Formulas:**\n$$\n\\begin{aligned}\nA_i &= \\frac{r_i - \\operatorname{mean}(r_{g(i)})}{\\operatorname{std}(r_{g(i)}) + \\texttt{eps}} \\qquad \\text{(unbiased std, as torch.std; TRL uses eps = 1e-4)} \\\\\n\\rho_{i,t} &= \\exp(\\text{logps}_{i,t} - \\text{old\\_logps}_{i,t}) \\\\\n\\text{KL}_{i,t} &= e^{\\,\\text{ref}_{i,t} - \\text{logps}_{i,t}} - (\\text{ref}_{i,t} - \\text{logps}_{i,t}) - 1 \\qquad (k_3) \\\\\n\\ell_{i,t} &= -\\Bigl[\\min\\bigl(\\rho_{i,t} A_i,\\ \\operatorname{clip}(\\rho_{i,t}, 1-\\epsilon, 1+\\epsilon) A_i\\bigr) - \\beta\\, \\text{KL}_{i,t}\\Bigr] \\\\\nL &= \\frac{1}{B}\\sum_i \\frac{\\sum_t m_{i,t}\\, \\ell_{i,t}}{\\sum_t m_{i,t}}\n\\end{aligned}\n$$\n\n**Constraints:**\n- Aggregate as above: mean over each response's own tokens, then mean over responses.\n- Gradients flow only into `logps`.\n- Padding positions hold arbitrary finite values and must not affect the result. Do not modify the inputs.",
    "description_zh": "实现 DeepSeekMath 论文中的 GRPO 损失：组内归一化的优势、逐 token 的 PPO 裁剪比率、写在损失里的对参考模型的 k3 KL 惩罚，以及按回答内 token 平均的聚合。在框架里它对应 TRL 的 `loss_type=\"grpo\"` 和 verl 的 `loss_agg_mode=\"seq-mean-token-mean\"`；两个框架现在的默认值都已换成别的聚合方式（token 平均的变体），TRL 还默认 `beta=0`。\n\n每个提示采样 $G$ 个回答。一个回答的优势是它的奖励在本组内做标准化的结果，这个回答的所有 token 共用它，因此不需要 critic。\n\n**签名:** `grpo_token_loss(logps, old_logps, ref_logps, rewards, mask, group_size, clip_ratio=0.2, kl_coef=0.04, eps=1e-6) -> Tensor`\n\n**参数:**\n- `logps` — 当前策略对采样 token 的对数概率 (B, T)\n- `old_logps` — 生成这批数据的策略给出的对数概率 (B, T)，常数\n- `ref_logps` — 冻结的参考模型给出的对数概率 (B, T)，常数\n- `rewards` — 每个回答一个标量奖励 (B,)，常数\n- `mask` — (B, T)，回答 token 为 1，padding 为 0（float 或 bool）；每行至少有一个 token\n- `group_size` — $G \\ge 2$；$B$ 是 $G$ 的倍数，同一提示的 $G$ 个回答占连续的行\n- `clip_ratio` — $\\epsilon$；`kl_coef` — $\\beta$；`eps` — 加在标准差上\n\n**返回:** 标量损失\n\n**公式:**\n$$\n\\begin{aligned}\nA_i &= \\frac{r_i - \\operatorname{mean}(r_{g(i)})}{\\operatorname{std}(r_{g(i)}) + \\texttt{eps}} \\qquad \\text{（无偏标准差，同 torch.std；TRL 取 eps = 1e-4）} \\\\\n\\rho_{i,t} &= \\exp(\\text{logps}_{i,t} - \\text{old\\_logps}_{i,t}) \\\\\n\\text{KL}_{i,t} &= e^{\\,\\text{ref}_{i,t} - \\text{logps}_{i,t}} - (\\text{ref}_{i,t} - \\text{logps}_{i,t}) - 1 \\qquad (k_3) \\\\\n\\ell_{i,t} &= -\\Bigl[\\min\\bigl(\\rho_{i,t} A_i,\\ \\operatorname{clip}(\\rho_{i,t}, 1-\\epsilon, 1+\\epsilon) A_i\\bigr) - \\beta\\, \\text{KL}_{i,t}\\Bigr] \\\\\nL &= \\frac{1}{B}\\sum_i \\frac{\\sum_t m_{i,t}\\, \\ell_{i,t}}{\\sum_t m_{i,t}}\n\\end{aligned}\n$$\n\n**约束:**\n- 按上式聚合：先在每个回答自己的 token 上取平均，再对回答取平均。\n- 梯度只流向 `logps`。\n- padding 位置可以是任意有限值，不得影响结果。不要修改输入。",
    "function_name": "grpo_token_loss",
    "hint": "r = rewards.view(-1, G)\nA = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1, 1)\nratio = exp(logps - old_logps)\nsurr = min(ratio * A, clamp(ratio, 1-e, 1+e) * A)\nd = ref_logps - logps;  kl = exp(d) - d - 1\ntok = -(surr - beta * kl)\nloss = ((tok * m).sum(1) / m.sum(1)).mean()",
    "hint_zh": "r = rewards.view(-1, G)\nA = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)).view(-1, 1)\nratio = exp(logps - old_logps)\nsurr = min(ratio * A, clamp(ratio, 1-e, 1+e) * A)\nd = ref_logps - logps;  kl = exp(d) - d - 1\ntok = -(surr - beta * kl)\nloss = ((tok * m).sum(1) / m.sum(1)).mean()",
    "tests": [
        {
            "name": "Returns a finite scalar with a gradient",
            "code": """
import torch
torch.manual_seed(0)
B, T, G = 4, 5, 2
logps = (torch.randn(B, T) * 0.2 - 1).requires_grad_(True)
out = {fn}(logps, logps.detach(), logps.detach() + 0.1, torch.randn(B), torch.ones(B, T), G)
assert isinstance(out, torch.Tensor) and out.dim() == 0, 'Loss must be a scalar tensor'
assert torch.isfinite(out), 'Loss should be finite'
assert out.requires_grad, 'Loss must carry a gradient'
""",
        },
        {
            "name": "Matches a reference with padding, clipping and KL",
            "code": """
import torch
torch.manual_seed(1)
G, P, T = 4, 3, 6
B = G * P
eps_c, beta, eps = 0.3, 0.04, 1e-6
lengths = torch.randint(1, T + 1, (B,))
mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
old_logps = torch.randn(B, T) * 0.5 - 2
logps = old_logps + torch.randn(B, T) * 0.4
ref_logps = old_logps + torch.randn(B, T) * 0.3
rewards = torch.randn(B) + torch.arange(P).repeat_interleave(G) * 3.0  # groups have very different means
out = {fn}(logps, old_logps, ref_logps, rewards, mask, G, clip_ratio=eps_c, kl_coef=beta, eps=eps)

# Independent reference: explicit loops over prompts, responses and tokens
seq_losses, n_clipped = [], 0
for p in range(P):
    r = rewards[p * G:(p + 1) * G]
    mu = r.sum() / G
    sd = (((r - mu) ** 2).sum() / (G - 1)).sqrt()
    for j in range(G):
        i = p * G + j
        a = (r[j] - mu) / (sd + eps)
        toks = []
        for t in range(int(lengths[i])):
            rho = torch.exp(logps[i, t] - old_logps[i, t])
            unclipped, clipped = rho * a, rho.clamp(1 - eps_c, 1 + eps_c) * a
            n_clipped += int(clipped < unclipped)
            surr = min(unclipped, clipped)
            d = ref_logps[i, t] - logps[i, t]
            kl = torch.exp(d) - d - 1
            toks.append(-(surr - beta * kl))
        seq_losses.append(torch.stack(toks).mean())
ref = torch.stack(seq_losses).mean()
assert n_clipped > 0, 'test setup: some tokens must be clipped'
assert torch.allclose(out, ref, atol=1e-5), f'got {out.item():.6f}, expected {ref.item():.6f}'
""",
        },
        {
            "name": "Gradient at ratio 1: -A_i / (B * n_i) per token",
            "code": """
import torch
# With logps == old_logps == ref_logps the KL term has zero gradient and the ratio is 1,
# so dL/dlogps_{i,t} = -A_i / (B * n_i): checks the unbiased std and the per-response mean
torch.manual_seed(6)
G, T = 3, 4
rewards = torch.tensor([1.0, 0.0, 0.0, 2.0, 5.0, 2.0])
B = rewards.numel()
lengths = [4, 1, 2, 3, 4, 2]
mask = torch.zeros(B, T)
for i, n in enumerate(lengths):
    mask[i, :n] = 1
base = torch.randn(B, T) * 0.3 - 1
logps = base.clone().requires_grad_(True)
{fn}(logps, base.clone(), base.clone(), rewards, mask, G).backward()
r = rewards.view(-1, G)
A = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1)
want = torch.zeros(B, T)
for i, n in enumerate(lengths):
    want[i, :n] = -A[i] / (B * n)
assert logps.grad is not None, 'No gradient for logps'
assert torch.allclose(logps.grad, want, atol=1e-6), f'grad max diff {(logps.grad - want).abs().max().item():.2e}; expected -A_i / (B * n_i) with the unbiased std'
""",
        },
        {
            "name": "KL term is the k3 estimator added as a penalty",
            "code": """
import torch
torch.manual_seed(2)
# ratio == 1 everywhere: within each group the advantages sum to 0 and every response is a
# per-sequence mean of a constant, so the policy part cancels and only beta * mean k3 remains
G, B, T, beta = 2, 4, 3, 0.1
logps = torch.randn(B, T) * 0.3 - 1
ref_logps = logps + torch.tensor([[0.5, -0.4, 0.3], [-0.6, 0.2, 0.1], [0.4, 0.4, -0.2], [0.0, -0.5, 0.6]])
mask = torch.ones(B, T)
rewards = torch.tensor([1.0, 0.0, 3.0, 2.0])
out = {fn}(logps, logps.clone(), ref_logps, rewards, mask, G, kl_coef=beta)
d = ref_logps - logps
k3 = torch.exp(d) - d - 1
want = beta * k3.mean(dim=1).mean()
assert torch.allclose(out, want, atol=1e-6), f'got {out.item():.6f}, expected {want.item():.6f} = beta * mean(k3); KL must be k3 = exp(ref-logp) - (ref-logp) - 1, entering the loss with a + sign'
""",
        },
        {
            "name": "Group with identical rewards has zero advantage",
            "code": """
import torch
torch.manual_seed(3)
G, T = 4, 3
rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
B = rewards.numel()
old = torch.randn(B, T) * 0.3
logps = (old + torch.randn(B, T) * 0.2).requires_grad_(True)
out = {fn}(logps, old, logps.detach().clone(), rewards, torch.ones(B, T), G, kl_coef=0.0)
assert torch.isfinite(out), 'std = 0 must not produce NaN or inf (eps)'
out.backward()
assert logps.grad[:G].abs().max() == 0, 'All-equal group should give zero advantage and no policy gradient'
assert logps.grad[G:].abs().max() > 0, 'Mixed group should receive a gradient'
""",
        },
        {
            "name": "Gradient reaches only logps",
            "code": """
import torch
torch.manual_seed(4)
G, B, T = 2, 4, 5
logps = (torch.randn(B, T) * 0.3).requires_grad_(True)
old = (logps.detach() + 0.2 * torch.randn(B, T)).requires_grad_(True)
ref = (logps.detach() + 0.2 * torch.randn(B, T)).requires_grad_(True)
rewards = torch.randn(B).requires_grad_(True)
mask = torch.ones(B, T); mask[1, 3:] = 0
{fn}(logps, old, ref, rewards, mask, G).backward()
assert logps.grad is not None and logps.grad.abs().max() > 0, 'logps must receive a gradient'
for name, t in [('old_logps', old), ('ref_logps', ref), ('rewards', rewards)]:
    assert t.grad is None or t.grad.abs().max() == 0, f'{name} must be treated as a constant (detach it)'
assert logps.grad[1, 3:].abs().max() == 0, 'Padding positions must get zero gradient'
""",
        },
        {
            "name": "Padding is ignored; bool mask works; inputs not mutated",
            "code": """
import torch
torch.manual_seed(5)
G, B, T = 3, 6, 5
mask = torch.ones(B, T, dtype=torch.bool)
mask[0, 2:] = False; mask[4, 1:] = False
args = [torch.randn(B, T) * 0.3 - 1 for _ in range(3)]
rewards = torch.randn(B)
saved = [a.clone() for a in args] + [rewards.clone(), mask.clone()]
out1 = {fn}(*args, rewards, mask, G)
for name, before, after in zip(['logps', 'old_logps', 'ref_logps', 'rewards', 'mask'], saved, args + [rewards, mask]):
    assert torch.equal(before, after), f'{name} was modified in place'
pad = ~mask
noisy = [a.masked_fill(pad, 0.0) + pad * torch.randn(B, T) for a in args]
out2 = {fn}(*noisy, rewards, mask.float(), G)
assert torch.allclose(out1, out2, atol=1e-6), 'Result depends on padding positions'
""",
        },
    ],
    "solution": '''def grpo_token_loss(logps, old_logps, ref_logps, rewards, mask, group_size,
                    clip_ratio=0.2, kl_coef=0.04, eps=1e-6):
    m = mask.to(logps.dtype)
    old_logps, ref_logps = old_logps.detach(), ref_logps.detach()

    # Group-relative advantage: standardize each reward within its own group (unbiased std)
    r = rewards.detach().to(logps.dtype).view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    adv = adv.view(-1, 1)                                  # (B, 1), shared by all tokens

    # PPO clipped surrogate, per token
    ratio = torch.exp(logps - old_logps)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv)

    # k3 estimate of KL(pi || pi_ref): non-negative, unbiased under samples from pi
    log_r = ref_logps - logps
    kl = torch.exp(log_r) - log_r - 1

    token_loss = -(surr - kl_coef * kl)
    # Mean over each response's own tokens, then over responses
    seq_loss = (token_loss * m).sum(dim=1) / m.sum(dim=1)
    return seq_loss.mean()''',
    "demo": """import torch

torch.manual_seed(0)
G, P, T = 4, 2, 6
B = G * P
lengths = torch.tensor([6, 3, 5, 2, 4, 6, 1, 3])
mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
old_logps = torch.randn(B, T) * 0.3 - 1.0
logps = (old_logps + torch.randn(B, T) * 0.1).requires_grad_(True)
ref_logps = old_logps + torch.randn(B, T) * 0.1
rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # second group: all correct

loss = grpo_token_loss(logps, old_logps, ref_logps, rewards, mask, G)
loss.backward()
print('loss:', round(loss.item(), 6))
print('grad norm per response:', [round(g, 4) for g in logps.grad.norm(dim=1).tolist()])
print('(the all-correct group only feels the KL term)')""",

}
