"""DPO-family preference losses: IPO, cDPO, SimPO, ORPO task."""

TASK = {
    "title": "DPO Variants: IPO / cDPO / SimPO / ORPO",
    "title_zh": "DPO 变体：IPO / cDPO / SimPO / ORPO",
    "difficulty": "Medium",
    "description_en": "Implement four DPO-family preference losses behind one function: IPO, cDPO (label-smoothed DPO), SimPO and ORPO. They train on the same paired data as DPO (a chosen and a rejected response per prompt) and differ in the shape of the loss, whether a reference model is needed, and whether log-probs are length-normalized.\n\n**Signature:** `preference_loss(chosen_logps, rejected_logps, chosen_lens, rejected_lens, method, ref_chosen_logps=None, ref_rejected_logps=None, beta=0.1, label_smoothing=0.0, gamma=0.5) -> Tensor`\n\n**Parameters:**\n- `chosen_logps`, `rejected_logps` — policy log-probs of each response, summed over its tokens (B,)\n- `chosen_lens`, `rejected_lens` — number of response tokens (B,), integer or float tensors\n- `method` — `\"ipo\"`, `\"cdpo\"`, `\"simpo\"` or `\"orpo\"`\n- `ref_chosen_logps`, `ref_rejected_logps` — reference-model summed log-probs (B,); required by `ipo` and `cdpo`, ignored by `simpo` and `orpo`\n- `beta` — the method's scale $\\beta$ (for ORPO, the weight $\\lambda$ of the odds-ratio term)\n- `label_smoothing` — $\\varepsilon$ for cDPO, the assumed probability that a preference label is flipped\n- `gamma` — SimPO's target margin $\\gamma$\n\n**Returns:** scalar loss, the mean over the batch\n\n**Formulas** (with $h = (\\pi_c - \\mathrm{ref}_c) - (\\pi_r - \\mathrm{ref}_r)$ and $\\bar a = \\pi / |y|$ the average log-prob per token):\n- IPO: $\\operatorname{mean}\\bigl[(h - \\tfrac{1}{2\\beta})^2\\bigr]$\n- cDPO: $z = \\beta h$, $\\operatorname{mean}\\bigl[-(1-\\varepsilon)\\log\\sigma(z) - \\varepsilon\\log\\sigma(-z)\\bigr]$\n- SimPO: $z = \\beta(\\bar a_c - \\bar a_r) - \\gamma$, $\\operatorname{mean}\\bigl[-\\log\\sigma(z)\\bigr]$\n- ORPO: $\\operatorname{lo}(\\bar a) = \\bar a - \\log(1 - e^{\\bar a})$ (the log-odds of $p = e^{\\bar a}$), $L = \\operatorname{mean}[-\\bar a_c] + \\beta\\,\\operatorname{mean}\\bigl[-\\log\\sigma(\\operatorname{lo}(\\bar a_c) - \\operatorname{lo}(\\bar a_r))\\bigr]$\n\n**Constraints:**\n- Reference log-probs are constants: no gradient flows into them.\n- `ipo` or `cdpo` without both reference tensors → `ValueError`; an unknown `method` → `ValueError`.\n- Numerically stable: $\\log\\sigma$ must stay finite for $|z|$ up to $10^3$ (use `F.logsigmoid`), and $\\log(1 - e^{\\bar a})$ must stay finite when $\\bar a$ is close to 0, e.g. $\\bar a = -10^{-8}$ in float32 (compute it with `torch.expm1`).\n- Do not modify the inputs.",
    "description_zh": "用一个函数实现 DPO 家族的四种偏好损失：IPO、cDPO（带标签平滑的 DPO）、SimPO 和 ORPO。它们和 DPO 用同样的成对数据（每个提示一个选中回答、一个拒绝回答），区别在于损失的形状、是否需要参考模型、对数概率是否按长度归一化。\n\n**签名:** `preference_loss(chosen_logps, rejected_logps, chosen_lens, rejected_lens, method, ref_chosen_logps=None, ref_rejected_logps=None, beta=0.1, label_smoothing=0.0, gamma=0.5) -> Tensor`\n\n**参数:**\n- `chosen_logps`、`rejected_logps` — 策略模型对每个回答的对数概率，按 token 求和 (B,)\n- `chosen_lens`、`rejected_lens` — 回答的 token 数 (B,)，整数或浮点张量\n- `method` — `\"ipo\"`、`\"cdpo\"`、`\"simpo\"` 或 `\"orpo\"`\n- `ref_chosen_logps`、`ref_rejected_logps` — 参考模型的序列对数概率 (B,)；`ipo` 和 `cdpo` 必须提供，`simpo` 和 `orpo` 忽略它们\n- `beta` — 各方法的缩放系数 $\\beta$（ORPO 中是赔率比项的权重 $\\lambda$）\n- `label_smoothing` — cDPO 的 $\\varepsilon$，即假设偏好标签被标反的概率\n- `gamma` — SimPO 的目标间隔 $\\gamma$\n\n**返回:** 标量损失，对 batch 取平均\n\n**公式**（记 $h = (\\pi_c - \\mathrm{ref}_c) - (\\pi_r - \\mathrm{ref}_r)$，$\\bar a = \\pi / |y|$ 为每个 token 的平均对数概率）：\n- IPO：$\\operatorname{mean}\\bigl[(h - \\tfrac{1}{2\\beta})^2\\bigr]$\n- cDPO：$z = \\beta h$，$\\operatorname{mean}\\bigl[-(1-\\varepsilon)\\log\\sigma(z) - \\varepsilon\\log\\sigma(-z)\\bigr]$\n- SimPO：$z = \\beta(\\bar a_c - \\bar a_r) - \\gamma$，$\\operatorname{mean}\\bigl[-\\log\\sigma(z)\\bigr]$\n- ORPO：$\\operatorname{lo}(\\bar a) = \\bar a - \\log(1 - e^{\\bar a})$（$p = e^{\\bar a}$ 的对数赔率），$L = \\operatorname{mean}[-\\bar a_c] + \\beta\\,\\operatorname{mean}\\bigl[-\\log\\sigma(\\operatorname{lo}(\\bar a_c) - \\operatorname{lo}(\\bar a_r))\\bigr]$\n\n**约束:**\n- 参考模型的对数概率是常数，梯度不流入它们。\n- `ipo` 或 `cdpo` 缺少任一参考张量时抛出 `ValueError`；未知的 `method` 抛出 `ValueError`。\n- 数值稳定：$|z|$ 达到 $10^3$ 时 $\\log\\sigma$ 仍须有限（用 `F.logsigmoid`）；$\\bar a$ 接近 0 时（例如 float32 下 $\\bar a = -10^{-8}$）$\\log(1 - e^{\\bar a})$ 仍须有限（用 `torch.expm1` 计算）。\n- 不要修改输入。",
    "function_name": "preference_loss",
    "hint": "if method in ('ipo', 'cdpo'):\n  h = (c - ref_c) - (r - ref_r)      # refs detached\n  ipo:  ((h - 1/(2*beta))**2).mean()\n  cdpo: z = beta*h; (-(1-eps)*logsigmoid(z) - eps*logsigmoid(-z)).mean()\na_c, a_r = c / len_c, r / len_r       # average log-prob\nsimpo: -logsigmoid(beta*(a_c - a_r) - gamma).mean()\norpo:  lo(a) = a - log(-expm1(a))\n       -a_c.mean() + beta * -logsigmoid(lo(a_c) - lo(a_r)).mean()",
    "hint_zh": "if method in ('ipo', 'cdpo'):\n  h = (c - ref_c) - (r - ref_r)      # 参考值 detach\n  ipo:  ((h - 1/(2*beta))**2).mean()\n  cdpo: z = beta*h; (-(1-eps)*logsigmoid(z) - eps*logsigmoid(-z)).mean()\na_c, a_r = c / len_c, r / len_r       # 平均对数概率\nsimpo: -logsigmoid(beta*(a_c - a_r) - gamma).mean()\norpo:  lo(a) = a - log(-expm1(a))\n       -a_c.mean() + beta * -logsigmoid(lo(a_c) - lo(a_r)).mean()",
    "tests": [
        {
            "name": "Every method returns a finite scalar with a gradient",
            "code": """
import torch
torch.manual_seed(0)
B = 4
c = (torch.rand(B) * -20 - 1).requires_grad_(True)
r = (torch.rand(B) * -20 - 1).requires_grad_(True)
lc = torch.randint(5, 30, (B,))
lr = torch.randint(5, 30, (B,))
rc = c.detach() + torch.randn(B)
rr = r.detach() + torch.randn(B)
for m in ['ipo', 'cdpo', 'simpo', 'orpo']:
    out = {fn}(c, r, lc, lr, m, rc, rr)
    assert isinstance(out, torch.Tensor) and out.dim() == 0, f'{m}: loss must be a scalar tensor'
    assert torch.isfinite(out), f'{m}: loss should be finite, got {out}'
    assert out.requires_grad, f'{m}: loss must carry a gradient'
""",
        },
        {
            "name": "IPO matches the squared loss and is zero at the target margin",
            "code": """
import torch
torch.manual_seed(1)
B = 6
c = torch.randn(B) * 3 - 20
r = torch.randn(B) * 3 - 20
rc = c + torch.randn(B)
rr = r + torch.randn(B)
lc = torch.randint(5, 30, (B,))
lr = torch.randint(5, 30, (B,))
beta = 0.3
h = (c - rc) - (r - rr)
ref = ((h - 1 / (2 * beta)) ** 2).mean()
out = {fn}(c, r, lc, lr, 'ipo', rc, rr, beta=beta)
assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), f'IPO: got {out.item():.6f}, expected {ref.item():.6f}'
# When the margin h equals 1/(2*beta) the IPO loss is exactly zero
r0 = torch.tensor([-5.0])
c0 = r0 + 1 / (2 * beta)
z0 = torch.zeros(1)
out0 = {fn}(c0, r0, torch.tensor([3]), torch.tensor([4]), 'ipo', z0, z0, beta=beta)
assert abs(out0.item()) < 1e-5, f'IPO loss at h = 1/(2*beta) should be 0, got {out0.item():.6f}'
""",
        },
        {
            "name": "cDPO uses label smoothing, reduces to DPO at 0, and is stable at large margins",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(2)
B = 6
c = torch.randn(B) * 5 - 30
r = torch.randn(B) * 5 - 30
rc = c + torch.randn(B) * 2
rr = r + torch.randn(B) * 2
lc = torch.randint(5, 30, (B,))
lr = torch.randint(5, 30, (B,))
beta, eps = 0.3, 0.2
z = beta * ((c - rc) - (r - rr))
ref = (-(1 - eps) * F.logsigmoid(z) - eps * F.logsigmoid(-z)).mean()
out = {fn}(c, r, lc, lr, 'cdpo', rc, rr, beta=beta, label_smoothing=eps)
assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), f'cDPO eps=0.2: got {out.item():.6f}, expected {ref.item():.6f}'
# Default label_smoothing=0 is plain DPO
ref_dpo = -F.logsigmoid(z).mean()
out_dpo = {fn}(c, r, lc, lr, 'cdpo', rc, rr, beta=beta)
assert torch.allclose(out_dpo, ref_dpo, atol=1e-5, rtol=1e-5), f'cDPO eps=0 should equal DPO: got {out_dpo.item():.6f}, expected {ref_dpo.item():.6f}'
# Margins of +-600: log(sigmoid(z)) underflows to -inf, logsigmoid does not
c2 = torch.tensor([0.0, -2000.0])
r2 = torch.tensor([-2000.0, 0.0])
zero = torch.zeros(2)
z2 = 0.3 * (c2 - r2)
ref2 = (-(0.9) * F.logsigmoid(z2) - 0.1 * F.logsigmoid(-z2)).mean()
out2 = {fn}(c2, r2, torch.tensor([10, 10]), torch.tensor([10, 10]), 'cdpo', zero, zero, beta=0.3, label_smoothing=0.1)
assert torch.isfinite(out2), f'cDPO must stay finite at |z| = 600, got {out2.item()}'
assert torch.allclose(out2, ref2, rtol=1e-5), f'cDPO at |z| = 600: got {out2.item():.4f}, expected {ref2.item():.4f}'
""",
        },
        {
            "name": "SimPO length-normalizes, subtracts the margin, and ignores the reference",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(3)
B = 6
lc = torch.tensor([5, 40, 12, 8, 30, 3])
lr = torch.tensor([25, 6, 12, 50, 9, 20])
c = -(torch.rand(B) * 0.8 + 0.2) * lc
r = -(torch.rand(B) * 0.8 + 0.2) * lr
beta, gamma = 2.0, 0.8
a_c = c / lc
a_r = r / lr
ref = -F.logsigmoid(beta * (a_c - a_r) - gamma).mean()
out = {fn}(c, r, lc, lr, 'simpo', beta=beta, gamma=gamma)
assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), f'SimPO: got {out.item():.6f}, expected {ref.item():.6f}'
# Passing reference log-probs must not change SimPO
rc = c + torch.randn(B) * 3
rr = r + torch.randn(B) * 3
out2 = {fn}(c, r, lc, lr, 'simpo', rc, rr, beta=beta, gamma=gamma)
assert torch.allclose(out2, ref, atol=1e-5, rtol=1e-5), 'SimPO uses no reference model; passing one changed the loss'
# Float lengths work too
out3 = {fn}(c, r, lc.float(), lr.float(), 'simpo', beta=beta, gamma=gamma)
assert torch.allclose(out3, ref, atol=1e-5, rtol=1e-5), 'Float lengths gave a different SimPO loss'
""",
        },
        {
            "name": "ORPO matches the reference, including an average log-prob near 0",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(4)
B = 6
lc = torch.tensor([5, 40, 12, 8, 30, 3])
lr = torch.tensor([25, 6, 12, 50, 9, 20])
c = -(torch.rand(B) * 1.5 + 0.05) * lc
r = -(torch.rand(B) * 1.5 + 0.05) * lr
beta = 0.25

def ref_orpo(c, r, lc, lr, beta):
    a_c = c.double() / lc.double()
    a_r = r.double() / lr.double()
    lo = lambda a: a - torch.log(-torch.expm1(a))
    return (-a_c).mean() + beta * (-F.logsigmoid(lo(a_c) - lo(a_r))).mean()

out = {fn}(c, r, lc, lr, 'orpo', beta=beta)
ref = ref_orpo(c, r, lc, lr, beta)
assert abs(out.item() - ref.item()) < 1e-4, f'ORPO: got {out.item():.6f}, expected {ref.item():.6f}'
# Average log-prob -1e-8 in float32 (chosen in row 0, rejected in row 1): exp(a) rounds to 1.0,
# so log(1 - exp(a)) is -inf and the rejected row's log-odds blow up
c1 = torch.tensor([-1e-8, -3.0])
r1 = torch.tensor([-2.0, -1e-8])
lc1 = torch.tensor([1, 2])
lr1 = torch.tensor([1, 1])
out1 = {fn}(c1, r1, lc1, lr1, 'orpo', beta=beta)
ref1 = ref_orpo(c1, r1, lc1, lr1, beta)
assert torch.isfinite(out1), f'ORPO must stay finite when the average log-prob is near 0, got {out1.item()}'
assert abs(out1.item() - ref1.item()) < 1e-4, f'ORPO near 0: got {out1.item():.6f}, expected {ref1.item():.6f}'
""",
        },
        {
            "name": "Gradients match for every method; reference log-probs get none",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(5)
B = 5
lc = torch.tensor([5, 40, 12, 8, 30])
lr = torch.tensor([25, 6, 12, 50, 9])
c0 = -(torch.rand(B) * 1.5 + 0.05) * lc
r0 = -(torch.rand(B) * 1.5 + 0.05) * lr
rc0 = c0 + torch.randn(B)
rr0 = r0 + torch.randn(B)
eps, gamma = 0.1, 0.8
betas = {'ipo': 0.3, 'cdpo': 0.3, 'simpo': 2.0, 'orpo': 0.7}

def ref_loss(m, c, r, beta):
    if m in ('ipo', 'cdpo'):
        h = (c - rc0) - (r - rr0)
        if m == 'ipo':
            return ((h - 1 / (2 * beta)) ** 2).mean()
        z = beta * h
        return (-(1 - eps) * F.logsigmoid(z) - eps * F.logsigmoid(-z)).mean()
    a_c, a_r = c / lc, r / lr
    if m == 'simpo':
        return -F.logsigmoid(beta * (a_c - a_r) - gamma).mean()
    lo = lambda a: a - torch.log(-torch.expm1(a))
    return (-a_c).mean() + beta * (-F.logsigmoid(lo(a_c) - lo(a_r))).mean()

for m, beta in betas.items():
    c = c0.clone().requires_grad_(True)
    r = r0.clone().requires_grad_(True)
    rc = rc0.clone().requires_grad_(True)
    rr = rr0.clone().requires_grad_(True)
    {fn}(c, r, lc, lr, m, rc, rr, beta=beta, label_smoothing=eps, gamma=gamma).backward()
    assert rc.grad is None or torch.all(rc.grad == 0), f'{m}: gradient leaked into ref_chosen_logps'
    assert rr.grad is None or torch.all(rr.grad == 0), f'{m}: gradient leaked into ref_rejected_logps'
    assert c.grad is not None and r.grad is not None, f'{m}: no gradient reached the policy log-probs'
    c2 = c0.clone().requires_grad_(True)
    r2 = r0.clone().requires_grad_(True)
    ref_loss(m, c2, r2, beta).backward()
    assert torch.allclose(c.grad, c2.grad, atol=1e-5, rtol=1e-4), f'{m}: chosen gradient differs from the reference'
    assert torch.allclose(r.grad, r2.grad, atol=1e-5, rtol=1e-4), f'{m}: rejected gradient differs from the reference'
""",
        },
        {
            "name": "Defaults are beta=0.1, label_smoothing=0, gamma=0.5",
            "code": """
import torch
import torch.nn.functional as F
torch.manual_seed(6)
B = 6
lc = torch.tensor([5, 40, 12, 8, 30, 3])
lr = torch.tensor([25, 6, 12, 50, 9, 20])
c = -(torch.rand(B) * 1.5 + 0.05) * lc
r = -(torch.rand(B) * 1.5 + 0.05) * lr
rc = c + torch.randn(B) * 2
rr = r + torch.randn(B) * 2
h = (c - rc) - (r - rr)
a_c, a_r = c / lc, r / lr
lo = lambda a: a - torch.log(-torch.expm1(a))
expected = {
    'ipo': ((h - 1 / (2 * 0.1)) ** 2).mean(),
    'cdpo': -F.logsigmoid(0.1 * h).mean(),
    'simpo': -F.logsigmoid(0.1 * (a_c - a_r) - 0.5).mean(),
    'orpo': (-a_c).mean() + 0.1 * (-F.logsigmoid(lo(a_c) - lo(a_r))).mean(),
}
for m, ref in expected.items():
    out = {fn}(c, r, lc, lr, m, rc, rr)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), f'{m} with default arguments: got {out.item():.6f}, expected {ref.item():.6f}'
""",
        },
        {
            "name": "Missing reference or unknown method raises ValueError",
            "code": """
import torch
c = torch.tensor([-5.0, -6.0])
r = torch.tensor([-7.0, -4.0])
lens = torch.tensor([3, 4])
ref = torch.tensor([-5.5, -6.5])
cases = [
    ('ipo', None, None),
    ('cdpo', None, None),
    ('cdpo', ref, None),
    ('ipo', None, ref),
    ('dpo_plus', ref, ref),
]
for m, rc, rr in cases:
    try:
        {fn}(c, r, lens, lens, m, rc, rr)
    except ValueError:
        continue
    raise AssertionError(f'Expected ValueError for method={m!r} with refs given: {rc is not None}, {rr is not None}')
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(7)
B = 4
c = torch.rand(B) * -20 - 1
r = torch.rand(B) * -20 - 1
lc = torch.randint(5, 30, (B,))
lr = torch.randint(5, 30, (B,))
rc = c + torch.randn(B)
rr = r + torch.randn(B)
saved = [t.clone() for t in (c, r, lc, lr, rc, rr)]
for m in ['ipo', 'cdpo', 'simpo', 'orpo']:
    {fn}(c, r, lc, lr, m, rc, rr, beta=0.3, label_smoothing=0.1, gamma=0.8)
for name, t, t0 in zip(['chosen_logps', 'rejected_logps', 'chosen_lens', 'rejected_lens', 'ref_chosen_logps', 'ref_rejected_logps'], (c, r, lc, lr, rc, rr), saved):
    assert torch.equal(t, t0), f'{name} was modified in place'
""",
        },
    ],
    "solution": '''def preference_loss(chosen_logps, rejected_logps, chosen_lens, rejected_lens, method,
                    ref_chosen_logps=None, ref_rejected_logps=None,
                    beta=0.1, label_smoothing=0.0, gamma=0.5):
    if method in ('ipo', 'cdpo'):
        if ref_chosen_logps is None or ref_rejected_logps is None:
            raise ValueError(f'{method} needs both reference log-probs')
        # Implicit reward margin, as in DPO; the reference model is a constant
        h = ((chosen_logps - ref_chosen_logps.detach())
             - (rejected_logps - ref_rejected_logps.detach()))
        if method == 'ipo':
            # Regress the margin to 1/(2*beta) instead of pushing it to infinity
            return ((h - 1 / (2 * beta)) ** 2).mean()
        z = beta * h
        eps = label_smoothing
        # The label is flipped with probability eps
        return (-(1 - eps) * F.logsigmoid(z) - eps * F.logsigmoid(-z)).mean()
    if method not in ('simpo', 'orpo'):
        raise ValueError(f'unknown method: {method!r}')

    # Reference-free methods use the average log-prob per token
    a_c = chosen_logps / chosen_lens.to(chosen_logps.dtype)
    a_r = rejected_logps / rejected_lens.to(rejected_logps.dtype)
    if method == 'simpo':
        return -F.logsigmoid(beta * (a_c - a_r) - gamma).mean()

    def log_odds(a):
        # log(p / (1 - p)) with p = exp(a); expm1 keeps 1 - p accurate when p is near 1
        return a - torch.log(-torch.expm1(a))

    nll = -a_c.mean()
    return nll + beta * (-F.logsigmoid(log_odds(a_c) - log_odds(a_r))).mean()''',
    "demo": """import torch

torch.manual_seed(0)
B = 4
chosen_lens = torch.tensor([12, 30, 8, 20])
rejected_lens = torch.tensor([25, 10, 40, 20])
chosen = -0.6 * chosen_lens + torch.randn(B)
rejected = -0.9 * rejected_lens + torch.randn(B)
ref_chosen = chosen - 1.0
ref_rejected = rejected + 0.5

print('IPO  :', round(preference_loss(chosen, rejected, chosen_lens, rejected_lens, 'ipo', ref_chosen, ref_rejected, beta=0.1).item(), 4))
print('cDPO :', round(preference_loss(chosen, rejected, chosen_lens, rejected_lens, 'cdpo', ref_chosen, ref_rejected, beta=0.1, label_smoothing=0.1).item(), 4))
print('SimPO:', round(preference_loss(chosen, rejected, chosen_lens, rejected_lens, 'simpo', beta=2.0, gamma=0.5).item(), 4))
print('ORPO :', round(preference_loss(chosen, rejected, chosen_lens, rejected_lens, 'orpo', beta=0.1).item(), 4))""",

}
