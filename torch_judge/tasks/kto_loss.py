"""KTO (Kahneman-Tversky Optimization) Loss task."""

TASK = {
    "title": "KTO (Kahneman-Tversky Optimization) Loss",
    "title_zh": "KTO 损失",
    "difficulty": "Medium",
    "description_en": "Implement the KTO loss. KTO needs no preference pairs: each completion is only labeled desirable or undesirable. It compares the implicit reward $r = \\log\\pi_\\theta(y \\mid x) - \\log\\pi_{\\text{ref}}(y \\mid x)$ with a reference point $z_0$, a batch estimate of $\\operatorname{KL}(\\pi_\\theta \\Vert \\pi_{\\text{ref}})$, and pushes desirable completions above it and undesirable ones below it.\n\n**Signature:** `kto_loss(policy_logps, ref_logps, desirable, kl_policy_logps, kl_ref_logps, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0) -> Tensor`\n\n**Parameters:**\n- `policy_logps`, `ref_logps` — summed log-probs of each completion under the policy and the reference model (B,)\n- `desirable` — bool (B,), True for desirable completions\n- `kl_policy_logps`, `kl_ref_logps` — summed log-probs (M,) of mismatched (prompt, completion) pairs under the policy and the reference model, used only to estimate $z_0$\n- `beta` — scale $\\beta$\n- `desirable_weight`, `undesirable_weight` — $\\lambda_D$ and $\\lambda_U$\n\n**Returns:** scalar loss\n\n**Formulas:**\n$$\n\\begin{aligned}\nr_i &= \\pi_i - \\mathrm{ref}_i \\\\\nz_0 &= \\max\\bigl(0,\\ \\operatorname{mean}(\\texttt{kl\\_policy\\_logps} - \\texttt{kl\\_ref\\_logps})\\bigr) \\\\\n\\ell_i &= \\begin{cases} \\lambda_D\\,\\bigl(1 - \\sigma(\\beta(r_i - z_0))\\bigr) & \\text{desirable} \\\\ \\lambda_U\\,\\bigl(1 - \\sigma(\\beta(z_0 - r_i))\\bigr) & \\text{undesirable} \\end{cases} \\\\\nL &= \\frac{1}{B}\\sum_i \\ell_i\n\\end{aligned}\n$$\n\n**Constraints:**\n- $z_0$ is a constant: no gradient flows into `kl_policy_logps` or `kl_ref_logps`.\n- `ref_logps` is a constant too; gradients flow only into `policy_logps`.\n- A batch may be all desirable or all undesirable.\n- Do not modify the inputs.",
    "description_zh": "实现 KTO 损失。KTO 不需要成对的偏好数据：每个回答只标注「好」（desirable）或「坏」（undesirable）。它把隐式奖励 $r = \\log\\pi_\\theta(y \\mid x) - \\log\\pi_{\\text{ref}}(y \\mid x)$ 与参考点 $z_0$ 比较（$z_0$ 是 $\\operatorname{KL}(\\pi_\\theta \\Vert \\pi_{\\text{ref}})$ 在一个 batch 上的估计），把好回答推到参考点之上，把坏回答压到参考点之下。\n\n**签名:** `kto_loss(policy_logps, ref_logps, desirable, kl_policy_logps, kl_ref_logps, beta=0.1, desirable_weight=1.0, undesirable_weight=1.0) -> Tensor`\n\n**参数:**\n- `policy_logps`、`ref_logps` — 每个回答在策略模型和参考模型下的序列对数概率（按 token 求和）(B,)\n- `desirable` — bool (B,)，True 表示好回答\n- `kl_policy_logps`、`kl_ref_logps` — 错配的（提示, 回答）对在策略模型和参考模型下的序列对数概率 (M,)，只用来估计 $z_0$\n- `beta` — 缩放系数 $\\beta$\n- `desirable_weight`、`undesirable_weight` — $\\lambda_D$ 和 $\\lambda_U$\n\n**返回:** 标量损失\n\n**公式:**\n$$\n\\begin{aligned}\nr_i &= \\pi_i - \\mathrm{ref}_i \\\\\nz_0 &= \\max\\bigl(0,\\ \\operatorname{mean}(\\texttt{kl\\_policy\\_logps} - \\texttt{kl\\_ref\\_logps})\\bigr) \\\\\n\\ell_i &= \\begin{cases} \\lambda_D\\,\\bigl(1 - \\sigma(\\beta(r_i - z_0))\\bigr) & \\text{好回答} \\\\ \\lambda_U\\,\\bigl(1 - \\sigma(\\beta(z_0 - r_i))\\bigr) & \\text{坏回答} \\end{cases} \\\\\nL &= \\frac{1}{B}\\sum_i \\ell_i\n\\end{aligned}\n$$\n\n**约束:**\n- $z_0$ 是常数：梯度不流入 `kl_policy_logps` 和 `kl_ref_logps`。\n- `ref_logps` 也是常数，梯度只流入 `policy_logps`。\n- 一个 batch 可能全是好回答或全是坏回答。\n- 不要修改输入。",
    "function_name": "kto_loss",
    "hint": "z0 = (kl_policy - kl_ref).mean().clamp(min=0).detach()\nr = policy_logps - ref_logps.detach()\ngood = w_D * (1 - sigmoid(beta * (r - z0)))\nbad  = w_U * (1 - sigmoid(beta * (z0 - r)))\nloss = where(desirable, good, bad).mean()",
    "hint_zh": "z0 = (kl_policy - kl_ref).mean().clamp(min=0).detach()\nr = policy_logps - ref_logps.detach()\ngood = w_D * (1 - sigmoid(beta * (r - z0)))\nbad  = w_U * (1 - sigmoid(beta * (z0 - r)))\nloss = where(desirable, good, bad).mean()",
    "tests": [
        {
            "name": "Returns a finite scalar with a gradient",
            "code": """
import torch
torch.manual_seed(0)
B, M = 6, 6
pol = (torch.randn(B) * 3 - 30).requires_grad_(True)
ref = pol.detach() + torch.randn(B)
des = torch.tensor([True, False, True, True, False, False])
klp = torch.randn(M) * 3 - 40
klr = klp + torch.randn(M)
out = {fn}(pol, ref, des, klp, klr)
assert isinstance(out, torch.Tensor) and out.dim() == 0, 'Loss must be a scalar tensor'
assert torch.isfinite(out), 'Loss should be finite'
assert out.requires_grad, 'Loss must carry a gradient'
""",
        },
        {
            "name": "Matches the reference on a mixed batch with non-default weights",
            "code": """
import torch
torch.manual_seed(1)
B, M = 8, 8
pol = torch.randn(B) * 3 - 30
ref = pol + torch.randn(B) * 2
des = torch.tensor([True, False, True, True, False, False, True, False])
klr = torch.randn(M) * 3 - 40
klp = klr + torch.randn(M) + 1.5          # positive KL estimate
beta, wd, wu = 0.3, 1.5, 0.7
z0 = (klp - klr).mean().clamp(min=0)
assert z0 > 0.5
r = pol - ref
good = wd * (1 - torch.sigmoid(beta * (r - z0)))
bad = wu * (1 - torch.sigmoid(beta * (z0 - r)))
expected = torch.where(des, good, bad).mean()
out = {fn}(pol, ref, des, klp, klr, beta=beta, desirable_weight=wd, undesirable_weight=wu)
assert torch.allclose(out, expected, atol=1e-6), f'got {out.item():.6f}, expected {expected.item():.6f}'
""",
        },
        {
            "name": "The reference point is clamped at zero",
            "code": """
import torch
torch.manual_seed(2)
B, M = 6, 5
pol = torch.randn(B) * 3 - 30
ref = pol + torch.randn(B) * 2
des = torch.tensor([True, False, True, False, True, False])
klr = torch.randn(M) * 3 - 40
klp = klr - 2.0 + torch.randn(M) * 0.3    # the KL estimate comes out negative
assert (klp - klr).mean() < -1
beta = 0.5
r = pol - ref
expected = torch.where(des, 1 - torch.sigmoid(beta * r), 1 - torch.sigmoid(-beta * r)).mean()
out = {fn}(pol, ref, des, klp, klr, beta=beta)
assert torch.allclose(out, expected, atol=1e-6), f'A negative KL estimate must give z0 = 0: got {out.item():.6f}, expected {expected.item():.6f}'
""",
        },
        {
            "name": "Gradient reaches only policy_logps, with the right sign per class",
            "code": """
import torch
torch.manual_seed(3)
B, M = 6, 6
pol0 = torch.randn(B) * 3 - 30
ref0 = pol0 + torch.randn(B) * 2
des = torch.tensor([True, False, True, False, True, False])
klr0 = torch.randn(M) * 3 - 40
klp0 = klr0 + torch.randn(M) + 1.0
beta, wd, wu = 0.3, 1.5, 0.7
pol = pol0.clone().requires_grad_(True)
ref = ref0.clone().requires_grad_(True)
klp = klp0.clone().requires_grad_(True)
klr = klr0.clone().requires_grad_(True)
{fn}(pol, ref, des, klp, klr, beta=beta, desirable_weight=wd, undesirable_weight=wu).backward()
for name, t in [('ref_logps', ref), ('kl_policy_logps', klp), ('kl_ref_logps', klr)]:
    assert t.grad is None or torch.all(t.grad == 0), f'Gradient leaked into {name}; it must be a constant'
assert pol.grad is not None, 'No gradient reached policy_logps'
# Reference gradient with z0 and the reference model held constant
pol2 = pol0.clone().requires_grad_(True)
z0 = (klp0 - klr0).mean().clamp(min=0)
r = pol2 - ref0
torch.where(des, wd * (1 - torch.sigmoid(beta * (r - z0))), wu * (1 - torch.sigmoid(beta * (z0 - r)))).mean().backward()
assert torch.allclose(pol.grad, pol2.grad, atol=1e-6), f'policy gradient differs: {pol.grad} vs {pol2.grad}'
# Raising a desirable completion's log-prob lowers the loss; raising an undesirable one's raises it
assert torch.all(pol.grad[des] < 0), 'Desirable completions should have negative gradients'
assert torch.all(pol.grad[~des] > 0), 'Undesirable completions should have positive gradients'
""",
        },
        {
            "name": "All-desirable and all-undesirable batches",
            "code": """
import torch
torch.manual_seed(4)
B, M = 5, 4
pol = torch.randn(B) * 3 - 30
ref = pol + torch.randn(B) * 2
klr = torch.randn(M) * 3 - 40
klp = klr + torch.randn(M) + 1.0
z0 = (klp - klr).mean().clamp(min=0)
beta, wd, wu = 0.3, 2.0, 0.5
r = pol - ref
all_good = torch.ones(B, dtype=torch.bool)
exp_good = (wd * (1 - torch.sigmoid(beta * (r - z0)))).mean()
out_good = {fn}(pol, ref, all_good, klp, klr, beta=beta, desirable_weight=wd, undesirable_weight=wu)
assert torch.isfinite(out_good) and torch.allclose(out_good, exp_good, atol=1e-6), f'All desirable: got {out_good.item():.6f}, expected {exp_good.item():.6f}'
all_bad = torch.zeros(B, dtype=torch.bool)
exp_bad = (wu * (1 - torch.sigmoid(beta * (z0 - r)))).mean()
out_bad = {fn}(pol, ref, all_bad, klp, klr, beta=beta, desirable_weight=wd, undesirable_weight=wu)
assert torch.isfinite(out_bad) and torch.allclose(out_bad, exp_bad, atol=1e-6), f'All undesirable: got {out_bad.item():.6f}, expected {exp_bad.item():.6f}'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(5)
B, M = 6, 6
pol = torch.randn(B) * 3 - 30
ref = pol + torch.randn(B)
des = torch.tensor([True, False, True, True, False, False])
klp = torch.randn(M) * 3 - 40
klr = klp + torch.randn(M)
saved = [t.clone() for t in (pol, ref, des, klp, klr)]
{fn}(pol, ref, des, klp, klr, beta=0.3, desirable_weight=1.5, undesirable_weight=0.7)
for name, t, t0 in zip(['policy_logps', 'ref_logps', 'desirable', 'kl_policy_logps', 'kl_ref_logps'], (pol, ref, des, klp, klr), saved):
    assert torch.equal(t, t0), f'{name} was modified in place'
""",
        },
    ],
    "solution": '''def kto_loss(policy_logps, ref_logps, desirable, kl_policy_logps, kl_ref_logps,
             beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    # Reference point: a batch estimate of KL(policy || ref), clamped at 0 and not trained through
    z0 = (kl_policy_logps - kl_ref_logps).mean().clamp(min=0).detach()
    r = policy_logps - ref_logps.detach()          # implicit reward
    # Desirable: push r above z0; undesirable: push r below z0
    good = desirable_weight * (1 - torch.sigmoid(beta * (r - z0)))
    bad = undesirable_weight * (1 - torch.sigmoid(beta * (z0 - r)))
    return torch.where(desirable.bool(), good, bad).mean()''',
    "demo": """import torch

torch.manual_seed(0)
B = 6
ref_logps = torch.randn(B) * 3 - 30
policy_logps = ref_logps + torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0, 0.5])
desirable = torch.tensor([True, True, True, False, False, False])
kl_ref = torch.randn(B) * 3 - 40
kl_policy = kl_ref + 0.3

loss = kto_loss(policy_logps, ref_logps, desirable, kl_policy, kl_ref, beta=0.5)
print('z0 (KL estimate):', round((kl_policy - kl_ref).mean().clamp(min=0).item(), 4))
print('implicit rewards:', (policy_logps - ref_logps).tolist())
print('KTO loss:', round(loss.item(), 4))""",

}
