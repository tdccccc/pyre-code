"""KL Estimators k1 / k2 / k3 task."""

TASK = {
    "title": "KL Estimators k1 / k2 / k3",
    "title_zh": "KL 估计器 k1 / k2 / k3",
    "difficulty": "Easy",
    "description_en": "Estimate the per-token KL divergence between the policy and the reference model from sampled tokens.\n\nIn RLHF the exact KL over the whole vocabulary is never computed: only the log-probs of the tokens that were actually sampled are available. Schulman's three estimators turn one sample into an estimate of $\\operatorname{KL}(\\pi \\,\\|\\, \\pi_{\\text{ref}})$. PPO-style RLHF puts k1 into the reward; GRPO puts k3 into the loss.\n\n**Signature:** `kl_estimate(logps, ref_logps, kind=\"k3\") -> Tensor`\n\n**Parameters:**\n- `logps` — policy log-probs of the sampled tokens, any shape (e.g. (B, T))\n- `ref_logps` — reference-model log-probs of the same tokens, same shape\n- `kind` — `\"k1\"`, `\"k2\"` or `\"k3\"`\n\n**Returns:** per-token estimates, same shape as `logps`\n\n**Formulas** (tokens were sampled from the policy $\\pi$):\n$$\n\\begin{aligned}\nr &= \\log \\pi_{\\text{ref}}(x) - \\log \\pi(x) \\\\\nk_1 &= -r \\\\\nk_2 &= \\tfrac{1}{2} r^2 \\\\\nk_3 &= e^{r} - 1 - r\n\\end{aligned}\n$$\n\n**Constraints:**\n- Any other `kind` raises `ValueError`.\n- `ref_logps` is a constant: gradients flow only into `logps`.\n- Do not modify the inputs.",
    "description_zh": "用采样得到的 token 逐个估计策略与参考模型之间的 KL 散度。\n\nRLHF 里从不在整个词表上计算精确的 KL：手里只有实际采样出的那个 token 的对数概率。Schulman 提出的三种估计器用一个样本给出 $\\operatorname{KL}(\\pi \\,\\|\\, \\pi_{\\text{ref}})$ 的估计。PPO 式的 RLHF 把 k1 放进奖励，GRPO 把 k3 放进损失。\n\n**签名:** `kl_estimate(logps, ref_logps, kind=\"k3\") -> Tensor`\n\n**参数:**\n- `logps` — 策略对已采样 token 的对数概率，任意形状（如 (B, T)）\n- `ref_logps` — 参考模型对同一批 token 的对数概率，形状相同\n- `kind` — `\"k1\"`、`\"k2\"` 或 `\"k3\"`\n\n**返回:** 逐 token 的估计值，形状与 `logps` 相同\n\n**公式**（token 由策略 $\\pi$ 采样得到）：\n$$\n\\begin{aligned}\nr &= \\log \\pi_{\\text{ref}}(x) - \\log \\pi(x) \\\\\nk_1 &= -r \\\\\nk_2 &= \\tfrac{1}{2} r^2 \\\\\nk_3 &= e^{r} - 1 - r\n\\end{aligned}\n$$\n\n**约束:**\n- 其他 `kind` 抛出 `ValueError`。\n- `ref_logps` 视为常数：梯度只流向 `logps`。\n- 不得修改输入。",
    "function_name": "kl_estimate",
    "hint": "r = ref_logps.detach() - logps\nk1: -r\nk2: 0.5 * r ** 2\nk3: torch.exp(r) - 1 - r\nelse: raise ValueError",
    "hint_zh": "r = ref_logps.detach() - logps\nk1：-r\nk2：0.5 * r ** 2\nk3：torch.exp(r) - 1 - r\n其他：raise ValueError",
    "tests": [
        {
            "name": "Exact formulas for k1, k2, k3",
            "code": """
import torch
torch.manual_seed(0)
logps = -torch.rand(3, 5) * 4
ref = -torch.rand(3, 5) * 4
r = ref - logps
expected = {'k1': -r, 'k2': 0.5 * r ** 2, 'k3': torch.exp(r) - 1 - r}
for kind, exp in expected.items():
    out = {fn}(logps, ref, kind=kind)
    assert out.shape == logps.shape, f'{kind}: expected shape {tuple(logps.shape)}, got {tuple(out.shape)}'
    assert torch.allclose(out, exp, atol=1e-6), f'{kind}: max diff {(out - exp).abs().max().item():.2e}'
""",
        },
        {
            "name": "Default kind is k3",
            "code": """
import torch
logps = torch.tensor([-1.0, -2.0, -0.5])
ref = torch.tensor([-1.5, -1.0, -0.5])
r = ref - logps
out = {fn}(logps, ref)
assert torch.allclose(out, torch.exp(r) - 1 - r, atol=1e-6), f'Default should be k3, got {out.tolist()}'
""",
        },
        {
            "name": "k2 and k3 are non-negative, k1 can be negative",
            "code": """
import torch
torch.manual_seed(1)
logps = -torch.rand(200) * 5
ref = -torch.rand(200) * 5
k1 = {fn}(logps, ref, kind='k1')
k2 = {fn}(logps, ref, kind='k2')
k3 = {fn}(logps, ref, kind='k3')
assert (k2 >= 0).all(), 'k2 must be non-negative'
assert (k3 >= -1e-7).all(), f'k3 must be non-negative, min {k3.min().item():.3e}'
assert (k1 < 0).any(), 'k1 = log(pi/pi_ref) is negative whenever pi_ref(x) > pi(x)'
same = {fn}(logps, logps.clone(), kind='k3')
assert torch.allclose(same, torch.zeros_like(same), atol=1e-7), 'k3 must be 0 when the two models agree'
""",
        },
        {
            "name": "Monte-Carlo: k1 and k3 are unbiased for KL(pi || pi_ref)",
            "code": """
import torch
torch.manual_seed(2)
V = 6
p = torch.softmax(torch.randn(V), 0)        # policy
q = torch.softmax(torch.randn(V), 0)        # reference
true_kl = (p * (p.log() - q.log())).sum()
x = torch.multinomial(p, 400000, replacement=True)
logps, ref = p.log()[x], q.log()[x]
k1 = {fn}(logps, ref, kind='k1').mean()
k3 = {fn}(logps, ref, kind='k3').mean()
assert abs(k1 - true_kl) < 0.01, f'k1 mean {k1.item():.4f} vs true KL {true_kl.item():.4f}'
assert abs(k3 - true_kl) < 0.01, f'k3 mean {k3.item():.4f} vs true KL {true_kl.item():.4f}'
# When the two distributions are close, k2 is a good (slightly biased) estimate too
q2 = torch.softmax(p.log() + 0.05 * torch.randn(V), 0)
true_kl2 = (p * (p.log() - q2.log())).sum()
k2 = {fn}(p.log()[x], q2.log()[x], kind='k2').mean()
assert abs(k2 - true_kl2) < 0.2 * true_kl2 + 1e-4, f'k2 mean {k2.item():.6f} vs true KL {true_kl2.item():.6f}'
""",
        },
        {
            "name": "Gradient flows into logps only",
            "code": """
import torch
torch.manual_seed(3)
base = -torch.rand(4, 3) * 3
base_ref = -torch.rand(4, 3) * 3
for kind in ['k1', 'k2', 'k3']:
    logps = base.clone().requires_grad_(True)
    ref = base_ref.clone().requires_grad_(True)
    {fn}(logps, ref, kind=kind).sum().backward()
    l2 = base.clone().requires_grad_(True)
    r = base_ref - l2
    exp = {'k1': -r, 'k2': 0.5 * r ** 2, 'k3': torch.exp(r) - 1 - r}[kind]
    exp.sum().backward()
    assert logps.grad is not None, f'{kind}: no gradient for logps'
    assert torch.allclose(logps.grad, l2.grad, atol=1e-6), f'{kind}: wrong gradient for logps'
    assert ref.grad is None or torch.all(ref.grad == 0), f'{kind}: ref_logps must be treated as a constant'
""",
        },
        {
            "name": "Unknown kind raises ValueError; inputs not mutated",
            "code": """
import torch
logps = torch.tensor([-1.0, -2.0])
ref = torch.tensor([-1.5, -0.5])
l0, r0 = logps.clone(), ref.clone()
for kind in ['k1', 'k2', 'k3']:
    {fn}(logps, ref, kind=kind)
assert torch.equal(logps, l0) and torch.equal(ref, r0), 'Inputs were modified in place'
try:
    {fn}(logps, ref, kind='k4')
except ValueError:
    pass
else:
    raise AssertionError('kind="k4" should raise ValueError')
""",
        },
    ],
    "solution": '''def kl_estimate(logps, ref_logps, kind="k3"):
    # log(pi_ref / pi) of the sampled token; the reference model is a constant
    r = ref_logps.detach() - logps
    if kind == "k1":
        return -r
    if kind == "k2":
        return 0.5 * r ** 2
    if kind == "k3":
        # e^r - 1 - r >= 0 for every r, and E_pi[e^r] = 1 keeps it unbiased
        return torch.exp(r) - 1 - r
    raise ValueError(f"unknown kind: {kind}")''',
    "demo": """import torch

torch.manual_seed(0)
p = torch.softmax(torch.randn(6), 0)   # policy
q = torch.softmax(torch.randn(6), 0)   # reference
true_kl = (p * (p.log() - q.log())).sum().item()

x = torch.multinomial(p, 100000, replacement=True)
for kind in ['k1', 'k2', 'k3']:
    est = kl_estimate(p.log()[x], q.log()[x], kind=kind)
    print(f'{kind}: mean {est.mean().item():.4f}  std {est.std().item():.4f}  min {est.min().item():+.3f}')
print('true KL:', round(true_kl, 4))""",

}
