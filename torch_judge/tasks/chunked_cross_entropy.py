"""Chunked LM-Head Cross-Entropy task."""

TASK = {
    "title": "Chunked LM-Head Cross-Entropy",
    "title_zh": "分块 LM Head 交叉熵",
    "difficulty": "Medium",
    "description_en": "Compute the language-model loss without materializing the full logits.\n\nThe LM head turns (N, D) hidden states into (N, V) logits. With a vocabulary of 100k+ tokens and thousands of positions, that single tensor dominates activation memory. Chunked or fused cross-entropy (Liger-kernel, Apple's cut cross-entropy) processes rows in chunks, so only one (chunk, V) block of logits exists at a time.\n\n**Signature:** `chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100) -> Tensor`\n\n**Parameters:**\n- `hidden` — final hidden states (N, D)\n- `weight` — LM head weight (V, D); logits = `hidden @ weight.T` (no bias)\n- `targets` — class indices (N,), int64; rows whose target equals `ignore_index` are ignored\n- `chunk_size` — number of rows processed per chunk (positive int; need not divide N)\n- `ignore_index` — target value to exclude (default -100)\n\n**Returns:** scalar tensor, the mean negative log-likelihood over the non-ignored rows\n\n**Formula:**\n```\nloss_i = logsumexp(logits_i) - logits_i[target_i]        for target_i != ignore_index\nL = sum_i loss_i / #{i : target_i != ignore_index}\n```\n\n**Constraints:**\n- Process rows in order, chunk by chunk: rows [0, c), [c, 2c), ... (the last chunk may be shorter).\n- Each chunk's logits have shape (at most chunk_size, V). When chunk_size < N, no tensor with N*V or more elements may be created.\n- Ignored rows contribute to neither the sum nor the denominator.\n- If every row is ignored, return a scalar tensor equal to 0.0.\n- Gradients must flow to `hidden` and `weight`.",
    "description_zh": "在不实例化完整 logits 的前提下计算语言模型损失。\n\nLM Head 会把 (N, D) 的隐状态变为 (N, V) 的 logits。词表有十万量级、序列有数千位置时，这一个张量就会主导激活显存。分块或融合的交叉熵（Liger-kernel、Apple 的 cut cross-entropy）按行分块处理，任意时刻只存在一个 (chunk, V) 的 logits 块。\n\n**签名:** `chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100) -> Tensor`\n\n**参数:**\n- `hidden` — 最终隐状态 (N, D)\n- `weight` — LM Head 权重 (V, D)；logits = `hidden @ weight.T`（无 bias）\n- `targets` — 类别索引 (N,)，int64；目标等于 `ignore_index` 的行被忽略\n- `chunk_size` — 每个分块处理的行数（正整数，不要求整除 N）\n- `ignore_index` — 需要排除的目标值（默认 -100）\n\n**返回:** 标量张量，即未被忽略行的平均负对数似然\n\n**公式:**\n```\nloss_i = logsumexp(logits_i) - logits_i[target_i]        当 target_i != ignore_index\nL = sum_i loss_i / #{i : target_i != ignore_index}\n```\n\n**约束:**\n- 按行顺序逐块处理：行 [0, c)、[c, 2c)、……（最后一块可以更短）。\n- 每个分块的 logits 形状为（至多 chunk_size, V）。当 chunk_size < N 时，不得创建元素数大于等于 N*V 的张量。\n- 被忽略的行既不计入求和，也不计入分母。\n- 若所有行都被忽略，返回值为 0.0 的标量张量。\n- 梯度必须回传到 `hidden` 和 `weight`。",
    "function_name": "chunked_cross_entropy",
    "hint": "total, count = 0, 0\nfor start in range(0, N, chunk_size):\n  h = hidden[start:start + chunk_size]          # (c, D)\n  t = targets[start:start + chunk_size]\n  valid = t != ignore_index\n  logits = h @ weight.T                         # (c, V) only this chunk\n  lse = logsumexp(logits, dim=-1)\n  tgt = logits.gather(1, t.masked_fill(~valid, 0)[:, None])[:, 0]\n  total = total + ((lse - tgt) * valid).sum(); count += valid.sum()\nreturn total / max(count, 1)",
    "hint_zh": "total, count = 0, 0\nfor start in range(0, N, chunk_size):\n  h = hidden[start:start + chunk_size]          # (c, D)\n  t = targets[start:start + chunk_size]\n  valid = t != ignore_index\n  logits = h @ weight.T                         # (c, V)，仅本块\n  lse = logsumexp(logits, dim=-1)\n  tgt = logits.gather(1, t.masked_fill(~valid, 0)[:, None])[:, 0]\n  total = total + ((lse - tgt) * valid).sum(); count += valid.sum()\nreturn total / max(count, 1)",
    "tests": [
        {
            "name": "Returns a finite scalar",
            "code": """
import torch
torch.manual_seed(0)
N, D, V = 10, 8, 20
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.randint(0, V, (N,))
out = {fn}(hidden, weight, targets, chunk_size=4)
assert out.dim() == 0, f'Expected a scalar, got shape {tuple(out.shape)}'
assert torch.isfinite(out), 'Loss should be finite'
""",
        },
        {
            "name": "Matches F.cross_entropy for many chunk sizes",
            "code": """
import torch
torch.manual_seed(1)
N, D, V = 10, 8, 20
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.randint(0, V, (N,))
ref = torch.nn.functional.cross_entropy(hidden @ weight.T, targets)
# 3, 4, 7 do not divide N=10; 10 and 16 and 100 are >= N
for cs in [1, 3, 4, 7, 10, 16, 100]:
    out = {fn}(hidden, weight, targets, chunk_size=cs)
    assert torch.allclose(out, ref, atol=1e-5), f'chunk_size={cs}: got {out.item():.6f}, expected {ref.item():.6f}'
""",
        },
        {
            "name": "Ignored rows excluded from sum and mean",
            "code": """
import torch
torch.manual_seed(2)
N, D, V = 12, 8, 10
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.randint(0, V, (N,))
targets[[0, 5, 6, 11]] = -100
# Independent reference: log_softmax over valid rows only, then mean
logits = hidden @ weight.T
valid = targets != -100
lp = torch.log_softmax(logits, dim=-1)
rows = torch.arange(N)
ref = -lp[rows[valid], targets[valid]].mean()
out = {fn}(hidden, weight, targets, chunk_size=5, ignore_index=-100)
assert torch.allclose(out, ref, atol=1e-5), f'Ignored rows leaked into the loss: got {out.item():.6f}, expected {ref.item():.6f}'
# Custom ignore_index: class 7 is a real class here, but rows targeting it must be skipped
targets2 = torch.randint(0, V, (N,))
targets2[[2, 3]] = 7
valid2 = targets2 != 7
ref2 = -torch.log_softmax(logits, dim=-1)[rows[valid2], targets2[valid2]].mean()
out2 = {fn}(hidden, weight, targets2, chunk_size=5, ignore_index=7)
assert torch.allclose(out2, ref2, atol=1e-5), f'Custom ignore_index=7: got {out2.item():.6f}, expected {ref2.item():.6f}'
""",
        },
        {
            "name": "All rows ignored returns zero",
            "code": """
import torch
torch.manual_seed(3)
N, D, V = 6, 4, 9
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.full((N,), -100, dtype=torch.long)
out = {fn}(hidden, weight, targets, chunk_size=4)
assert out.dim() == 0, f'Expected a scalar, got shape {tuple(out.shape)}'
assert out.item() == 0.0, f'All-ignored batch should give 0.0, got {out.item()}'
""",
        },
        {
            "name": "Gradients match F.cross_entropy",
            "code": """
import torch
torch.manual_seed(4)
N, D, V = 9, 6, 12
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.randint(0, V, (N,))
targets[4] = -100
h1 = hidden.clone().requires_grad_(True)
w1 = weight.clone().requires_grad_(True)
out = {fn}(h1, w1, targets, chunk_size=4)
out.backward()
h2 = hidden.clone().requires_grad_(True)
w2 = weight.clone().requires_grad_(True)
ref = torch.nn.functional.cross_entropy(h2 @ w2.T, targets, ignore_index=-100)
ref.backward()
assert h1.grad is not None, 'No gradient for hidden'
assert w1.grad is not None, 'No gradient for weight (LM head)'
assert torch.allclose(h1.grad, h2.grad, atol=1e-5), f'hidden grad max diff {(h1.grad - h2.grad).abs().max().item():.2e}'
assert torch.allclose(w1.grad, w2.grad, atol=1e-5), f'weight grad max diff {(w1.grad - w2.grad).abs().max().item():.2e}'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(5)
N, D, V = 8, 4, 11
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.randint(0, V, (N,))
targets[1] = -100
h0, w0, t0 = hidden.clone(), weight.clone(), targets.clone()
{fn}(hidden, weight, targets, chunk_size=3)
assert torch.equal(hidden, h0), 'hidden was modified in place'
assert torch.equal(weight, w0), 'weight was modified in place'
assert torch.equal(targets, t0), 'targets was modified in place'
""",
        },
        {
            "name": "Peak logits stay below N*V (chunked, not materialized)",
            "code": """
import torch
from torch.overrides import TorchFunctionMode

class _Recorder(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.max_numel = 0
    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.max_numel = max(self.max_numel, out.numel())
        return out

N, D, V = 64, 16, 500
torch.manual_seed(6)
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.randint(0, V, (N,))
with torch.no_grad():
    ref = torch.nn.functional.cross_entropy(hidden @ weight.T, targets)
    rec = _Recorder()
    with rec:
        out = {fn}(hidden, weight, targets, chunk_size=16)
assert torch.allclose(out, ref, atol=1e-5), f'Wrong loss: got {out.item():.6f}, expected {ref.item():.6f}'
assert rec.max_numel < N * V, f'Created a tensor with {rec.max_numel} elements; full logits would be {N * V}'
""",
        },
    ],
    "solution": '''def chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100):
    N = hidden.shape[0]
    total = hidden.new_zeros(())
    count = 0
    for start in range(0, N, chunk_size):
        h = hidden[start:start + chunk_size]          # (c, D)
        t = targets[start:start + chunk_size]         # (c,)
        valid = t != ignore_index
        # Only this chunk's logits exist: (c, V) instead of (N, V)
        logits = h @ weight.T
        lse = torch.logsumexp(logits, dim=-1)         # (c,)
        # Ignored targets may be out of range for gather; point them at 0 and mask out
        safe_t = t.masked_fill(~valid, 0)
        tgt_logit = logits.gather(1, safe_t[:, None])[:, 0]
        total = total + ((lse - tgt_logit) * valid).sum()
        count += int(valid.sum())
    return total / max(count, 1)''',
    "demo": """import torch

torch.manual_seed(0)
N, D, V = 64, 16, 500
hidden = torch.randn(N, D)
weight = torch.randn(V, D)
targets = torch.randint(0, V, (N,))
targets[:5] = -100

loss = chunked_cross_entropy(hidden, weight, targets, chunk_size=16)
ref = torch.nn.functional.cross_entropy(hidden @ weight.T, targets)
print('Chunked loss:', round(loss.item(), 6))
print('Reference   :', round(ref.item(), 6))
print('Peak logits per chunk:', 16 * V, 'elements vs full:', N * V)""",

}
