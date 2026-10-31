"""Loss aggregation modes for token-level RL losses task."""

TASK = {
    "title": "Loss Aggregation Modes (token-mean vs seq-mean)",
    "title_zh": "损失聚合方式",
    "difficulty": "Easy",
    "description_en": "Reduce a (B, T) matrix of per-token losses to one scalar, in the four ways RL frameworks offer.\n\nPPO, GRPO and their variants all compute a loss per token first. How those tokens are averaged decides how much each token, and so each response, weighs in the gradient. The choice is a named option in verl (`loss_agg_mode`); its default is `token-mean`. TRL exposes the same choices through `loss_type` (`\"grpo\"` is seq-mean-token-mean, `\"dr_grpo\"` is seq-mean-token-sum-norm, `\"dapo\"` is token-mean).\n\n**Signature:** `aggregate_loss(token_loss, mask, mode, norm_len=None) -> Tensor`\n\n**Parameters:**\n- `token_loss` — per-token loss (B, T)\n- `mask` — (B, T), 1 on response tokens, 0 on padding (float or bool); every row has at least one token\n- `mode` — one of `\"token-mean\"`, `\"seq-mean-token-mean\"`, `\"seq-mean-token-sum\"`, `\"seq-mean-token-sum-norm\"`\n- `norm_len` — the constant $L$ for `seq-mean-token-sum-norm`; when `None`, use $L = T$ (the padded length)\n\n**Returns:** scalar tensor\n\n**Formulas** ($m$ is the mask, $\\ell$ the token loss, $n_i = \\sum_t m_{i,t}$):\n$$\n\\begin{aligned}\n\\texttt{token-mean} &: \\frac{\\sum_{i,t} m_{i,t}\\,\\ell_{i,t}}{\\sum_{i,t} m_{i,t}} \\\\\n\\texttt{seq-mean-token-mean} &: \\frac{1}{B}\\sum_i \\frac{\\sum_t m_{i,t}\\,\\ell_{i,t}}{n_i} \\\\\n\\texttt{seq-mean-token-sum} &: \\frac{1}{B}\\sum_i \\sum_t m_{i,t}\\,\\ell_{i,t} \\\\\n\\texttt{seq-mean-token-sum-norm} &: \\frac{1}{B}\\sum_i \\frac{\\sum_t m_{i,t}\\,\\ell_{i,t}}{L}\n\\end{aligned}\n$$\n\n**Constraints:**\n- Any other `mode` raises `ValueError`.\n- Padding positions hold arbitrary finite values and must not affect the result or receive gradient.\n- Gradients flow into `token_loss`. Do not modify the inputs.",
    "description_zh": "把 (B, T) 的逐 token 损失归约成一个标量，实现 RL 框架提供的四种方式。\n\nPPO、GRPO 及其变体都先算出每个 token 的损失。这些 token 怎么平均，决定了每个 token、进而每个回答在梯度里占多大权重。verl 把它做成一个配置项（`loss_agg_mode`），默认是 `token-mean`。TRL 用 `loss_type` 提供同样的选择（`\"grpo\"` 对应 seq-mean-token-mean，`\"dr_grpo\"` 对应 seq-mean-token-sum-norm，`\"dapo\"` 对应 token-mean）。\n\n**签名:** `aggregate_loss(token_loss, mask, mode, norm_len=None) -> Tensor`\n\n**参数:**\n- `token_loss` — 逐 token 损失 (B, T)\n- `mask` — (B, T)，回答 token 为 1，padding 为 0（float 或 bool）；每行至少有一个 token\n- `mode` — `\"token-mean\"`、`\"seq-mean-token-mean\"`、`\"seq-mean-token-sum\"`、`\"seq-mean-token-sum-norm\"` 之一\n- `norm_len` — `seq-mean-token-sum-norm` 用的常数 $L$；为 `None` 时取 $L = T$（padding 后的长度）\n\n**返回:** 标量张量\n\n**公式**（$m$ 是掩码，$\\ell$ 是 token 损失，$n_i = \\sum_t m_{i,t}$）：\n$$\n\\begin{aligned}\n\\texttt{token-mean} &: \\frac{\\sum_{i,t} m_{i,t}\\,\\ell_{i,t}}{\\sum_{i,t} m_{i,t}} \\\\\n\\texttt{seq-mean-token-mean} &: \\frac{1}{B}\\sum_i \\frac{\\sum_t m_{i,t}\\,\\ell_{i,t}}{n_i} \\\\\n\\texttt{seq-mean-token-sum} &: \\frac{1}{B}\\sum_i \\sum_t m_{i,t}\\,\\ell_{i,t} \\\\\n\\texttt{seq-mean-token-sum-norm} &: \\frac{1}{B}\\sum_i \\frac{\\sum_t m_{i,t}\\,\\ell_{i,t}}{L}\n\\end{aligned}\n$$\n\n**约束:**\n- 其他 `mode` 抛出 `ValueError`。\n- padding 位置可以是任意有限值，不得影响结果，也不应得到梯度。\n- 梯度流向 `token_loss`。不要修改输入。",
    "function_name": "aggregate_loss",
    "hint": "m = mask.float();  per_seq = (token_loss * m).sum(1)   # (B,)\ntoken-mean:              per_seq.sum() / m.sum()\nseq-mean-token-mean:     (per_seq / m.sum(1)).mean()\nseq-mean-token-sum:      per_seq.mean()\nseq-mean-token-sum-norm: per_seq.mean() / (norm_len or T)",
    "hint_zh": "m = mask.float();  per_seq = (token_loss * m).sum(1)   # (B,)\ntoken-mean:              per_seq.sum() / m.sum()\nseq-mean-token-mean:     (per_seq / m.sum(1)).mean()\nseq-mean-token-sum:      per_seq.mean()\nseq-mean-token-sum-norm: per_seq.mean() / (norm_len 或 T)",
    "tests": [
        {
            "name": "Returns a scalar with a gradient for every mode",
            "code": """
import torch
torch.manual_seed(0)
B, T = 3, 5
loss = torch.randn(B, T).requires_grad_(True)
mask = torch.ones(B, T); mask[1, 2:] = 0
for mode in ['token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm']:
    out = {fn}(loss, mask, mode)
    assert isinstance(out, torch.Tensor) and out.dim() == 0, f'{mode}: expected a scalar tensor'
    assert torch.isfinite(out), f'{mode}: result should be finite'
    assert out.requires_grad, f'{mode}: result must carry a gradient'
""",
        },
        {
            "name": "Each mode matches an explicit loop",
            "code": """
import torch
torch.manual_seed(1)
B, T = 4, 7
lengths = [7, 2, 5, 1]
mask = torch.zeros(B, T)
for i, n in enumerate(lengths):
    mask[i, :n] = 1
loss = torch.randn(B, T)
rows = [[loss[i, t].item() for t in range(n)] for i, n in enumerate(lengths)]
flat = [x for r in rows for x in r]
want = {
    'token-mean': sum(flat) / len(flat),
    'seq-mean-token-mean': sum(sum(r) / len(r) for r in rows) / B,
    'seq-mean-token-sum': sum(sum(r) for r in rows) / B,
    'seq-mean-token-sum-norm': sum(sum(r) for r in rows) / B / T,
}
for mode, w in want.items():
    out = {fn}(loss, mask, mode).item()
    assert abs(out - w) < 1e-5, f'{mode}: got {out:.6f}, expected {w:.6f}'
""",
        },
        {
            "name": "Per-token gradient weights",
            "code": """
import torch
# d(result)/d(loss[i,t]) for a valid token: token-mean 1/N, seq-mean-token-mean 1/(B*n_i),
# seq-mean-token-sum 1/B, seq-mean-token-sum-norm 1/(B*L); 0 on padding in every mode
torch.manual_seed(2)
B, T, L = 3, 6, 10
lengths = [6, 1, 3]
mask = torch.zeros(B, T)
for i, n in enumerate(lengths):
    mask[i, :n] = 1
N = sum(lengths)
for mode in ['token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm']:
    loss = torch.randn(B, T).requires_grad_(True)
    {fn}(loss, mask, mode, norm_len=L).backward()
    want = torch.zeros(B, T)
    for i, n in enumerate(lengths):
        w = {'token-mean': 1 / N, 'seq-mean-token-mean': 1 / (B * n),
             'seq-mean-token-sum': 1 / B, 'seq-mean-token-sum-norm': 1 / (B * L)}[mode]
        want[i, :n] = w
    assert loss.grad is not None, f'{mode}: no gradient for token_loss'
    assert torch.allclose(loss.grad, want, atol=1e-7), f'{mode}: per-token gradient weights are wrong, got {loss.grad.tolist()}'
""",
        },
        {
            "name": "norm_len sets the constant; None means T",
            "code": """
import torch
torch.manual_seed(3)
B, T = 3, 4
loss = torch.randn(B, T)
mask = torch.ones(B, T); mask[0, 1:] = 0; mask[2, 3:] = 0
per_seq_mean = (loss * mask).sum(1).mean()
out_none = {fn}(loss, mask, 'seq-mean-token-sum-norm')
assert torch.allclose(out_none, per_seq_mean / T, atol=1e-6), f'norm_len=None should divide by T={T}: got {out_none.item():.6f}, expected {(per_seq_mean / T).item():.6f}'
out_12 = {fn}(loss, mask, 'seq-mean-token-sum-norm', norm_len=12)
assert torch.allclose(out_12, per_seq_mean / 12, atol=1e-6), f'norm_len=12 should divide by 12: got {out_12.item():.6f}, expected {(per_seq_mean / 12).item():.6f}'
# The constant does not depend on each row's own length
out_mean = {fn}(loss, mask, 'seq-mean-token-mean')
assert not torch.allclose(out_none, out_mean), 'seq-mean-token-sum-norm must divide by a constant, not by each row length'
# Every row shorter than T: L is still the padded width T, not the longest valid row
T2 = 6
loss2 = torch.randn(3, T2)
mask2 = torch.zeros(3, T2)
for i, n in enumerate([4, 2, 3]):
    mask2[i, :n] = 1
want2 = (loss2 * mask2).sum(1).mean() / T2
out2 = {fn}(loss2, mask2, 'seq-mean-token-sum-norm')
assert torch.allclose(out2, want2, atol=1e-6), f'norm_len=None should divide by the padded width T={T2} even when no row fills it: got {out2.item():.6f}, expected {want2.item():.6f}'
""",
        },
        {
            "name": "Unknown mode raises ValueError",
            "code": """
import torch
loss = torch.randn(2, 3)
mask = torch.ones(2, 3)
for bad in ['mean', 'token-sum', 'seq-mean']:
    try:
        {fn}(loss, mask, bad)
    except ValueError:
        continue
    raise AssertionError(f'mode={bad!r} should raise ValueError')
""",
        },
        {
            "name": "Padding is ignored; bool mask works; inputs not mutated",
            "code": """
import torch
torch.manual_seed(4)
B, T = 4, 6
mask = torch.zeros(B, T, dtype=torch.bool)
for i, n in enumerate([6, 2, 4, 1]):
    mask[i, :n] = True
loss = torch.randn(B, T)
l0, m0 = loss.clone(), mask.clone()
for mode in ['token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm']:
    out1 = {fn}(loss, mask, mode, norm_len=8)
    assert torch.equal(loss, l0) and torch.equal(mask, m0), f'{mode}: inputs were modified in place'
    noisy = torch.where(mask, loss, torch.randn(B, T) * 100)
    out2 = {fn}(noisy, mask.float(), mode, norm_len=8)
    assert torch.allclose(out1, out2, atol=1e-5), f'{mode}: result depends on padding positions'
""",
        },
    ],
    "solution": '''def aggregate_loss(token_loss, mask, mode, norm_len=None):
    m = mask.to(token_loss.dtype)
    # Sum each sequence's valid tokens; padding is zeroed out by the mask
    per_seq = (token_loss * m).sum(dim=1)                 # (B,)
    if mode == "token-mean":
        # Every token in the batch weighs the same
        return per_seq.sum() / m.sum()
    if mode == "seq-mean-token-mean":
        # Every sequence weighs the same; its tokens share that weight
        return (per_seq / m.sum(dim=1)).mean()
    if mode == "seq-mean-token-sum":
        return per_seq.mean()
    if mode == "seq-mean-token-sum-norm":
        # Dr. GRPO: divide by a constant, so per-token weight does not depend on length
        L = token_loss.shape[1] if norm_len is None else norm_len
        return per_seq.mean() / L
    raise ValueError(f"unknown mode: {mode}")''',
    "demo": """import torch

# Two responses with the same per-token loss 1.0: one long (8 tokens), one short (2 tokens)
B, T = 2, 8
loss = torch.ones(B, T, requires_grad=True)
mask = torch.zeros(B, T)
mask[0, :8] = 1
mask[1, :2] = 1

for mode in ['token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm']:
    loss.grad = None
    out = aggregate_loss(loss, mask, mode)
    out.backward()
    w_long, w_short = loss.grad[0, 0].item(), loss.grad[1, 0].item()
    print(f'{mode:26s} value={out.item():.4f}  weight per token: long={w_long:.4f} short={w_short:.4f}')""",

}
