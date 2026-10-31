"""Token-Level Distillation (Forward / Reverse KL / JSD) task."""

TASK = {
    "title": "Token-Level Distillation (Forward / Reverse KL / JSD)",
    "title_zh": "知识蒸馏损失：前向 KL / 反向 KL / JSD",
    "difficulty": "Medium",
    "description_en": "Train a student language model to match a teacher's next-token distribution at every token (token-level knowledge distillation).\n\nDistilling a large model into a smaller one is a standard post-training step. At each position the student and the teacher each give a distribution over the vocabulary, and the loss measures how far apart they are. The direction matters: **forward KL** pushes the student to cover every mode of the teacher, **reverse KL** lets it commit to the teacher's main mode. GKD (generalized knowledge distillation, used for on-policy distillation) interpolates between the two with a generalized Jensen–Shannon divergence.\n\n**Signature:** `distill_loss(student_logits, teacher_logits, mask, mode=\"forward\", temperature=1.0, beta=0.5) -> Tensor`\n\n**Parameters:**\n- `student_logits` — (B, T, V)\n- `teacher_logits` — (B, T, V), a constant: no gradient may reach it, even if it has `requires_grad=True`\n- `mask` — (B, T), float or bool, 1 on the tokens that count; at least one token counts\n- `mode` — `\"forward\"`, `\"reverse\"` or `\"jsd\"`\n- `temperature` — $\\tau > 0$, divides both logits\n- `beta` — $\\beta$, the JSD mixing weight, used only by `\"jsd\"`\n\n**Returns:** scalar tensor\n\n**Formula:** at every token let $p_S = \\operatorname{softmax}(s / \\tau)$ and $p_T = \\operatorname{softmax}(t / \\tau)$ over the vocabulary, and $\\operatorname{KL}(p \\Vert q) = \\sum_v p_v (\\log p_v - \\log q_v)$.\n- `\"forward\"`: $D = \\operatorname{KL}(p_T \\Vert p_S)$\n- `\"reverse\"`: $D = \\operatorname{KL}(p_S \\Vert p_T)$\n- `\"jsd\"`: $M = \\beta\\, p_T + (1 - \\beta)\\, p_S$, $\\;D = \\beta\\, \\operatorname{KL}(p_T \\Vert M) + (1 - \\beta)\\, \\operatorname{KL}(p_S \\Vert M)$\n\n$$\nL = \\tau^2 \\cdot \\frac{\\sum_{b,t} m_{b,t}\\, D_{b,t}}{\\sum_{b,t} m_{b,t}}\n$$\n\n(a masked token-mean over the whole batch, scaled by $\\tau^2$).\n\n**Constraints:**\n- Work in log space (`log_softmax`, and `logsumexp` for $\\log M$): logits of magnitude ~1000 must still give a finite loss.\n- An unknown `mode` raises `ValueError`; `mode=\"jsd\"` with $\\beta \\le 0$ or $\\beta \\ge 1$ raises `ValueError`.\n- Gradients flow only to `student_logits`. Inputs must not be modified.",
    "description_zh": "让学生语言模型在每个 token 上去拟合教师的下一个 token 分布（token 级知识蒸馏）。\n\n把大模型蒸馏成小模型是常见的后训练步骤。每个位置上，学生和教师各给出一个词表上的分布，损失衡量两者差多远。方向很关键：**前向 KL** 逼学生覆盖教师的所有峰，**反向 KL** 允许学生只抓住教师的主峰。GKD（广义知识蒸馏，在线蒸馏常用）用广义 Jensen–Shannon 散度在两者之间插值。\n\n**签名:** `distill_loss(student_logits, teacher_logits, mask, mode=\"forward\", temperature=1.0, beta=0.5) -> Tensor`\n\n**参数:**\n- `student_logits` — (B, T, V)\n- `teacher_logits` — (B, T, V)，常数：即使它 `requires_grad=True`，也不能有梯度流到它\n- `mask` — (B, T)，float 或 bool，计入的 token 处为 1；至少有一个 token 计入\n- `mode` — `\"forward\"`、`\"reverse\"` 或 `\"jsd\"`\n- `temperature` — $\\tau > 0$，两边的 logits 都除以它\n- `beta` — $\\beta$，JSD 的混合权重，只在 `\"jsd\"` 中使用\n\n**返回:** 标量张量\n\n**公式:** 每个 token 上记 $p_S = \\operatorname{softmax}(s / \\tau)$、$p_T = \\operatorname{softmax}(t / \\tau)$（在词表维度上），$\\operatorname{KL}(p \\Vert q) = \\sum_v p_v (\\log p_v - \\log q_v)$。\n- `\"forward\"`：$D = \\operatorname{KL}(p_T \\Vert p_S)$\n- `\"reverse\"`：$D = \\operatorname{KL}(p_S \\Vert p_T)$\n- `\"jsd\"`：$M = \\beta\\, p_T + (1 - \\beta)\\, p_S$，$\\;D = \\beta\\, \\operatorname{KL}(p_T \\Vert M) + (1 - \\beta)\\, \\operatorname{KL}(p_S \\Vert M)$\n\n$$\nL = \\tau^2 \\cdot \\frac{\\sum_{b,t} m_{b,t}\\, D_{b,t}}{\\sum_{b,t} m_{b,t}}\n$$\n\n（在整个 batch 上按掩码做 token 平均，再乘以 $\\tau^2$）。\n\n**约束:**\n- 在对数空间里计算（`log_softmax`，$\\log M$ 用 `logsumexp`）：logits 量级约 1000 时损失仍须有限。\n- 未知的 `mode` 抛出 `ValueError`；`mode=\"jsd\"` 且 $\\beta \\le 0$ 或 $\\beta \\ge 1$ 时抛出 `ValueError`。\n- 梯度只流向 `student_logits`。不得修改输入。",
    "function_name": "distill_loss",
    "hint": "log_ps = log_softmax(student / tau)\nlog_pt = log_softmax(teacher.detach() / tau)        # teacher is a constant\nforward: sum_v exp(log_pt) * (log_pt - log_ps)\nreverse: sum_v exp(log_ps) * (log_ps - log_pt)\njsd:     log_m = logsumexp(stack([log_pt + log(beta), log_ps + log(1 - beta)]), dim=0)\n         beta * KL(pt || m) + (1 - beta) * KL(ps || m)\nloss = tau**2 * (D * mask).sum() / mask.sum()",
    "hint_zh": "log_ps = log_softmax(student / tau)\nlog_pt = log_softmax(teacher.detach() / tau)        # 教师是常数\nforward：sum_v exp(log_pt) * (log_pt - log_ps)\nreverse：sum_v exp(log_ps) * (log_ps - log_pt)\njsd：    log_m = logsumexp(stack([log_pt + log(beta), log_ps + log(1 - beta)]), dim=0)\n         beta * KL(pt || m) + (1 - beta) * KL(ps || m)\nloss = tau**2 * (D * mask).sum() / mask.sum()",
    "tests": [
        {
            "name": "Returns a finite non-negative scalar for every mode",
            "code": """
import torch
torch.manual_seed(0)
B, T, V = 2, 5, 7
s, t = torch.randn(B, T, V), torch.randn(B, T, V)
m = torch.ones(B, T)
for mode in ['forward', 'reverse', 'jsd']:
    out = {fn}(s, t, m, mode=mode)
    assert out.dim() == 0, f'{mode}: expected a scalar, got shape {tuple(out.shape)}'
    assert torch.isfinite(out), f'{mode}: loss should be finite'
    assert out.item() >= -1e-6, f'{mode}: a divergence cannot be negative, got {out.item()}'
""",
        },
        {
            "name": "Defaults: forward KL, temperature 1, JSD beta 0.5",
            "code": """
import torch, math

def _ref(s, t, m, mode='forward', tau=1.0, beta=0.5):
    # Independent float64 reference written from the formula
    s, t, m = s.double(), t.double(), m.double()
    ps, pt = torch.softmax(s / tau, -1), torch.softmax(t / tau, -1)
    lps, lpt = torch.log_softmax(s / tau, -1), torch.log_softmax(t / tau, -1)
    if mode == 'forward':
        d = (pt * (lpt - lps)).sum(-1)
    elif mode == 'reverse':
        d = (ps * (lps - lpt)).sum(-1)
    else:
        # log M via logaddexp so that one-hot distributions stay finite
        lm = torch.logaddexp(lpt + math.log(beta), lps + math.log(1 - beta))
        d = beta * (pt * (lpt - lm)).sum(-1) + (1 - beta) * (ps * (lps - lm)).sum(-1)
    return tau ** 2 * (d * m).sum() / m.sum()

torch.manual_seed(1)
B, T, V = 2, 6, 9
s, t = torch.randn(B, T, V) * 2, torch.randn(B, T, V) * 2
m = torch.ones(B, T)
out = {fn}(s, t, m)
ref = _ref(s, t, m, 'forward', 1.0)
assert abs(out.item() - ref.item()) < 1e-5, f'Default call: got {out.item():.6f}, expected forward KL(p_T || p_S) at tau=1 = {ref.item():.6f}'
out_jsd = {fn}(s, t, m, mode='jsd')
ref_jsd = _ref(s, t, m, 'jsd', 1.0, 0.5)
assert abs(out_jsd.item() - ref_jsd.item()) < 1e-5, f'mode="jsd" with default beta: got {out_jsd.item():.6f}, expected {ref_jsd.item():.6f} (beta=0.5)'
""",
        },
        {
            "name": "Forward and reverse KL match the formula at non-default temperatures",
            "code": """
import torch, math

def _ref(s, t, m, mode='forward', tau=1.0, beta=0.5):
    # Independent float64 reference written from the formula
    s, t, m = s.double(), t.double(), m.double()
    ps, pt = torch.softmax(s / tau, -1), torch.softmax(t / tau, -1)
    lps, lpt = torch.log_softmax(s / tau, -1), torch.log_softmax(t / tau, -1)
    if mode == 'forward':
        d = (pt * (lpt - lps)).sum(-1)
    elif mode == 'reverse':
        d = (ps * (lps - lpt)).sum(-1)
    else:
        # log M via logaddexp so that one-hot distributions stay finite
        lm = torch.logaddexp(lpt + math.log(beta), lps + math.log(1 - beta))
        d = beta * (pt * (lpt - lm)).sum(-1) + (1 - beta) * (ps * (lps - lm)).sum(-1)
    return tau ** 2 * (d * m).sum() / m.sum()

torch.manual_seed(2)
B, T, V = 3, 4, 11
s, t = torch.randn(B, T, V) * 2, torch.randn(B, T, V) * 2
m = torch.ones(B, T)
for tau in [2.0, 0.5]:
    for mode in ['forward', 'reverse']:
        out = {fn}(s, t, m, mode=mode, temperature=tau)
        ref = _ref(s, t, m, mode, tau)
        assert abs(out.item() - ref.item()) < 1e-4 * max(1.0, abs(ref.item())), f'{mode}, temperature={tau}: got {out.item():.6f}, expected {ref.item():.6f} (did you scale by tau**2?)'
fwd = {fn}(s, t, m, mode='forward', temperature=2.0)
rev = {fn}(s, t, m, mode='reverse', temperature=2.0)
assert abs(fwd.item() - rev.item()) > 1e-3, 'Forward and reverse KL should differ on these inputs'
""",
        },
        {
            "name": "Generalized JSD matches the formula for a non-default beta",
            "code": """
import torch, math

def _ref(s, t, m, mode='forward', tau=1.0, beta=0.5):
    # Independent float64 reference written from the formula
    s, t, m = s.double(), t.double(), m.double()
    ps, pt = torch.softmax(s / tau, -1), torch.softmax(t / tau, -1)
    lps, lpt = torch.log_softmax(s / tau, -1), torch.log_softmax(t / tau, -1)
    if mode == 'forward':
        d = (pt * (lpt - lps)).sum(-1)
    elif mode == 'reverse':
        d = (ps * (lps - lpt)).sum(-1)
    else:
        # log M via logaddexp so that one-hot distributions stay finite
        lm = torch.logaddexp(lpt + math.log(beta), lps + math.log(1 - beta))
        d = beta * (pt * (lpt - lm)).sum(-1) + (1 - beta) * (ps * (lps - lm)).sum(-1)
    return tau ** 2 * (d * m).sum() / m.sum()

torch.manual_seed(3)
B, T, V = 2, 5, 8
s, t = torch.randn(B, T, V) * 2, torch.randn(B, T, V) * 2
m = torch.ones(B, T)
for beta, tau in [(0.3, 1.5), (0.8, 1.0)]:
    out = {fn}(s, t, m, mode='jsd', temperature=tau, beta=beta)
    ref = _ref(s, t, m, 'jsd', tau, beta)
    assert abs(out.item() - ref.item()) < 1e-5, f'jsd beta={beta}, temperature={tau}: got {out.item():.6f}, expected {ref.item():.6f}; M = beta * p_T + (1 - beta) * p_S'
# JSD at beta = 0.5 is symmetric in student and teacher and bounded by log 2
a = {fn}(s, t, m, mode='jsd', beta=0.5)
b = {fn}(t, s, m, mode='jsd', beta=0.5)
assert abs(a.item() - b.item()) < 1e-5, 'JSD with beta=0.5 should not change when student and teacher swap'
assert a.item() <= 0.6931472 + 1e-6, f'JSD with beta=0.5 is at most log 2, got {a.item()}'
""",
        },
        {
            "name": "Masked token-mean over the whole batch",
            "code": """
import torch, math

def _ref(s, t, m, mode='forward', tau=1.0, beta=0.5):
    # Independent float64 reference written from the formula
    s, t, m = s.double(), t.double(), m.double()
    ps, pt = torch.softmax(s / tau, -1), torch.softmax(t / tau, -1)
    lps, lpt = torch.log_softmax(s / tau, -1), torch.log_softmax(t / tau, -1)
    if mode == 'forward':
        d = (pt * (lpt - lps)).sum(-1)
    elif mode == 'reverse':
        d = (ps * (lps - lpt)).sum(-1)
    else:
        # log M via logaddexp so that one-hot distributions stay finite
        lm = torch.logaddexp(lpt + math.log(beta), lps + math.log(1 - beta))
        d = beta * (pt * (lpt - lm)).sum(-1) + (1 - beta) * (ps * (lps - lm)).sum(-1)
    return tau ** 2 * (d * m).sum() / m.sum()

torch.manual_seed(4)
B, T, V = 3, 6, 5
s, t = torch.randn(B, T, V), torch.randn(B, T, V)
# Padding positions hold large garbage logits that must not leak into the loss
s[0, 4:] = 40 * torch.randn(2, V)
t[0, 4:] = -40 * torch.randn(2, V)
s[2, 2:] = 40 * torch.randn(4, V)
m = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
for mode in ['forward', 'reverse', 'jsd']:
    out = {fn}(s, t, m, mode=mode, temperature=1.3, beta=0.4)
    ref = _ref(s, t, m, mode, 1.3, 0.4)
    assert abs(out.item() - ref.item()) < 1e-5, f'{mode}: got {out.item():.6f}, expected {ref.item():.6f}; average over all masked tokens of the batch (not per row)'
    out_bool = {fn}(s, t, m.bool(), mode=mode, temperature=1.3, beta=0.4)
    assert abs(out_bool.item() - out.item()) < 1e-6, f'{mode}: a bool mask should give the same loss as a float mask'
""",
        },
        {
            "name": "Gradients reach the student only, for every mode",
            "code": """
import torch
torch.manual_seed(5)
B, T, V = 2, 4, 6
s0, t0 = torch.randn(B, T, V), torch.randn(B, T, V)
m = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
tau, beta = 1.7, 0.35

def _ref_loss(s, t, mode):
    lps = torch.log_softmax(s / tau, -1)
    lpt = torch.log_softmax(t / tau, -1)
    ps, pt = lps.exp(), lpt.exp()
    if mode == 'forward':
        d = (pt * (lpt - lps)).sum(-1)
    elif mode == 'reverse':
        d = (ps * (lps - lpt)).sum(-1)
    else:
        lm = torch.log(beta * pt + (1 - beta) * ps)
        d = beta * (pt * (lpt - lm)).sum(-1) + (1 - beta) * (ps * (lps - lm)).sum(-1)
    return tau ** 2 * (d * m).sum() / m.sum()

for mode in ['forward', 'reverse', 'jsd']:
    s1 = s0.clone().requires_grad_(True)
    t1 = t0.clone().requires_grad_(True)
    {fn}(s1, t1, m, mode=mode, temperature=tau, beta=beta).backward()
    s2 = s0.clone().requires_grad_(True)
    _ref_loss(s2, t0, mode).backward()
    assert s1.grad is not None, f'{mode}: no gradient reached student_logits'
    assert torch.allclose(s1.grad, s2.grad, atol=1e-5), f'{mode}: student grad max diff {(s1.grad - s2.grad).abs().max().item():.2e}'
    assert t1.grad is None or torch.all(t1.grad == 0), f'{mode}: gradient leaked into teacher_logits (detach the teacher)'
""",
        },
        {
            "name": "Stable for large logits",
            "code": """
import torch, math

def _ref(s, t, m, mode='forward', tau=1.0, beta=0.5):
    # Independent float64 reference written from the formula
    s, t, m = s.double(), t.double(), m.double()
    ps, pt = torch.softmax(s / tau, -1), torch.softmax(t / tau, -1)
    lps, lpt = torch.log_softmax(s / tau, -1), torch.log_softmax(t / tau, -1)
    if mode == 'forward':
        d = (pt * (lpt - lps)).sum(-1)
    elif mode == 'reverse':
        d = (ps * (lps - lpt)).sum(-1)
    else:
        # log M via logaddexp so that one-hot distributions stay finite
        lm = torch.logaddexp(lpt + math.log(beta), lps + math.log(1 - beta))
        d = beta * (pt * (lpt - lm)).sum(-1) + (1 - beta) * (ps * (lps - lm)).sum(-1)
    return tau ** 2 * (d * m).sum() / m.sum()

torch.manual_seed(6)
B, T, V = 2, 3, 6
s, t = torch.randn(B, T, V) * 1000, torch.randn(B, T, V) * 1000
m = torch.ones(B, T)
for mode in ['forward', 'reverse', 'jsd']:
    out = {fn}(s, t, m, mode=mode)
    assert torch.isfinite(out), f'{mode}: loss is not finite for logits ~1e3; use log_softmax / logsumexp instead of log(softmax)'
    ref = _ref(s, t, m, mode)
    assert abs(out.item() - ref.item()) < 1e-3 * max(1.0, abs(ref.item())), f'{mode}: got {out.item():.4f}, expected {ref.item():.4f}'
""",
        },
        {
            "name": "Invalid arguments raise, inputs are not mutated",
            "code": """
import torch
torch.manual_seed(7)
s, t = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
m = torch.ones(2, 3)
s0, t0, m0 = s.clone(), t.clone(), m.clone()
{fn}(s, t, m, mode='jsd', temperature=2.0, beta=0.3)
assert torch.equal(s, s0) and torch.equal(t, t0) and torch.equal(m, m0), 'Inputs were modified in place'
for kwargs in [dict(mode='kl'), dict(mode='jsd', beta=0.0), dict(mode='jsd', beta=1.0)]:
    try:
        {fn}(s, t, m, **kwargs)
    except ValueError:
        continue
    raise AssertionError(f'Expected ValueError for {kwargs}')
""",
        },
    ],
    "solution": '''def distill_loss(student_logits, teacher_logits, mask, mode="forward", temperature=1.0, beta=0.5):
    if mode not in ("forward", "reverse", "jsd"):
        raise ValueError(f"unknown mode: {mode}")
    if mode == "jsd" and not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    log_ps = torch.log_softmax(student_logits / temperature, dim=-1)
    # The teacher is a constant: cut it out of the graph
    log_pt = torch.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    if mode == "forward":
        div = (log_pt.exp() * (log_pt - log_ps)).sum(-1)              # KL(p_T || p_S)
    elif mode == "reverse":
        div = (log_ps.exp() * (log_ps - log_pt)).sum(-1)              # KL(p_S || p_T)
    else:
        # log M = log(beta * p_T + (1 - beta) * p_S), computed in log space
        log_m = torch.logsumexp(torch.stack([log_pt + math.log(beta), log_ps + math.log(1 - beta)]), dim=0)
        div = (beta * (log_pt.exp() * (log_pt - log_m)).sum(-1)
               + (1 - beta) * (log_ps.exp() * (log_ps - log_m)).sum(-1))
    m = mask.to(div.dtype)
    return temperature ** 2 * (div * m).sum() / m.sum()''',
    "demo": """import torch

# A bimodal teacher over 4 tokens, and two students: one spreads over both modes, one picks a single mode
teacher = torch.log(torch.tensor([[[0.48, 0.48, 0.02, 0.02]]]))
spread = torch.log(torch.tensor([[[0.30, 0.30, 0.20, 0.20]]]))
one_mode = torch.log(torch.tensor([[[0.94, 0.02, 0.02, 0.02]]]))
mask = torch.ones(1, 1)

for name, student in [('spread  ', spread), ('one-mode', one_mode)]:
    fwd = distill_loss(student, teacher, mask, mode='forward').item()
    rev = distill_loss(student, teacher, mask, mode='reverse').item()
    jsd = distill_loss(student, teacher, mask, mode='jsd', beta=0.5).item()
    print(f'{name}  forward KL {fwd:.3f}   reverse KL {rev:.3f}   JSD {jsd:.3f}')
print('Forward KL prefers the spread student, reverse KL the one that commits to a mode.')""",

}
