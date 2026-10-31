"""SFT Loss with Assistant Mask & Packing task."""

TASK = {
    "title": "SFT Loss with Assistant Mask & Packing",
    "title_zh": "SFT 损失：助手掩码与样本打包",
    "difficulty": "Medium",
    "description_en": "Compute the supervised fine-tuning (SFT) loss on chat data: train only on what the assistant wrote, and never let one packed sample predict the next.\n\nA chat sample is rendered through a chat template into one token sequence: system prompt, user turns, assistant turns. Only assistant tokens are learning targets. To avoid wasting compute on padding, trainers **pack** several short samples into one row; the model must then not be trained to predict the first token of sample $k+1$ from the end of sample $k$.\n\n**Signature:** `sft_loss(logits, input_ids, assistant_mask, seq_ids, reduction=\"token\") -> Tensor`\n\n**Parameters:**\n- `logits` — model output (B, T, V); `logits[:, t]` predicts `input_ids[:, t+1]`\n- `input_ids` — token ids (B, T), int64\n- `assistant_mask` — (B, T), 1 on tokens written by the assistant (including its end-of-turn token), 0 elsewhere\n- `seq_ids` — (B, T), int64: index of the packed sample each token belongs to; −1 on padding\n- `reduction` — `\"token\"` or `\"sample\"`\n\n**Returns:** scalar tensor\n\n**Which targets count:** the target of position $t$ is `input_ids[:, t+1]`. It counts iff\n- `assistant_mask[:, t+1] == 1`, and\n- `seq_ids[:, t+1] == seq_ids[:, t]` (no prediction across a packing boundary), and\n- `seq_ids[:, t+1] != -1` (padding is never a target).\n\n**Reductions:** with $\\ell$ the negative log-likelihood of a counted target,\n- `\"token\"`: $\\dfrac{\\sum \\ell}{\\#\\text{counted targets}}$ over the whole batch\n- `\"sample\"`: for every sample (row, seq_id) with at least one counted target, the mean $\\ell$ over its targets; then the mean over those samples. The same seq_id in different rows is a different sample.\n- any other value raises `ValueError`\n\n**Constraints:**\n- If no target counts, return a scalar equal to 0.0 that is still connected to `logits`: after `backward()`, `logits.grad` is all zeros (not `None`).\n- Gradients must flow to `logits`; inputs must not be modified.",
    "description_zh": "在对话数据上计算监督微调（SFT）损失：只训练助手写的部分，并且打包在一起的样本之间不能互相预测。\n\n一条对话样本经过聊天模板渲染成一个 token 序列：系统提示词、用户轮次、助手轮次。只有助手的 token 是学习目标。为了不在填充上浪费算力，训练框架会把多个短样本**打包**（packing）进同一行；这时不能让模型用第 $k$ 个样本的结尾去预测第 $k+1$ 个样本的第一个 token。\n\n**签名:** `sft_loss(logits, input_ids, assistant_mask, seq_ids, reduction=\"token\") -> Tensor`\n\n**参数:**\n- `logits` — 模型输出 (B, T, V)；`logits[:, t]` 预测 `input_ids[:, t+1]`\n- `input_ids` — token id (B, T)，int64\n- `assistant_mask` — (B, T)，助手写的 token（包括它的轮次结束 token）处为 1，其余为 0\n- `seq_ids` — (B, T)，int64：每个 token 属于打包进来的第几个样本；填充处为 −1\n- `reduction` — `\"token\"` 或 `\"sample\"`\n\n**返回:** 标量张量\n\n**哪些目标计入：** 位置 $t$ 的目标是 `input_ids[:, t+1]`。当且仅当同时满足以下条件时计入：\n- `assistant_mask[:, t+1] == 1`；\n- `seq_ids[:, t+1] == seq_ids[:, t]`（不跨打包边界预测）；\n- `seq_ids[:, t+1] != -1`（填充永远不是目标）。\n\n**归约方式:** 记 $\\ell$ 为一个计入目标的负对数似然，\n- `\"token\"`：在整个 batch 上取 $\\dfrac{\\sum \\ell}{\\text{计入的目标数}}$\n- `\"sample\"`：对每个至少有一个计入目标的样本（行, seq_id），先求它所有目标的平均 $\\ell$，再对这些样本取平均。不同行里相同的 seq_id 是不同的样本。\n- 其他取值抛出 `ValueError`\n\n**约束:**\n- 如果没有任何目标计入，返回值为 0.0 的标量，并且仍然连在 `logits` 的计算图上：`backward()` 之后 `logits.grad` 全为 0（而不是 `None`）。\n- 梯度必须回传到 `logits`；不得修改输入。",
    "function_name": "sft_loss",
    "hint": "nll[b, t] = logsumexp(logits[b, t]) - logits[b, t, input_ids[b, t+1]]    # (B, T-1)\ncur, nxt = seq_ids[:, :-1], seq_ids[:, 1:]\nvalid = assistant_mask[:, 1:].bool() & (nxt == cur) & (nxt != -1)\n\"token\":  nll[valid].sum() / valid.sum()\n\"sample\": key = row * (seq_ids.max() + 1) + seq_id for valid targets\n          per-key sum and count (index_add_ / bincount), then (sum / count).mean()\nnothing valid: return logits.sum() * 0.0",
    "hint_zh": "nll[b, t] = logsumexp(logits[b, t]) - logits[b, t, input_ids[b, t+1]]    # (B, T-1)\ncur, nxt = seq_ids[:, :-1], seq_ids[:, 1:]\nvalid = assistant_mask[:, 1:].bool() & (nxt == cur) & (nxt != -1)\n\"token\"：nll[valid].sum() / valid.sum()\n\"sample\"：对计入的目标算 key = 行号 * (seq_ids.max() + 1) + seq_id\n          按 key 求和、计数（index_add_ / bincount），再 (sum / count).mean()\n没有计入的目标：return logits.sum() * 0.0",
    "tests": [
        {
            "name": "Returns a finite scalar",
            "code": """
import torch
torch.manual_seed(0)
B, T, V = 2, 8, 11
logits = torch.randn(B, T, V)
ids = torch.randint(0, V, (B, T))
am = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0, 0]])
sid = torch.tensor([[0] * 8, [0, 0, 0, 0, 0, -1, -1, -1]])
out = {fn}(logits, ids, am, sid)
assert out.dim() == 0, f'Expected a scalar, got shape {tuple(out.shape)}'
assert torch.isfinite(out), 'Loss should be finite'
""",
        },
        {
            "name": "One sample per row matches shifted cross-entropy on assistant tokens",
            "code": """
import torch
torch.manual_seed(1)
B, T, V = 3, 12, 13
logits = torch.randn(B, T, V)
ids = torch.randint(0, V, (B, T))
am = torch.zeros(B, T, dtype=torch.long)
sid = torch.zeros(B, T, dtype=torch.long)
# Row 0: system, user, assistant, user, assistant (two assistant turns)
am[0, 3:6] = 1
am[0, 8:12] = 1
# Rows 1 and 2: prompt, assistant, then right padding
am[1, 4:9] = 1
sid[1, 9:] = -1
am[2, 6:8] = 1
sid[2, 8:] = -1
# HF-style reference: labels = ids on assistant tokens, -100 elsewhere, shifted by one
labels = torch.where(am.bool() & (sid >= 0), ids, torch.full_like(ids, -100))
ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
out = {fn}(logits, ids, am, sid)
assert torch.allclose(out, ref, atol=1e-5), f'got {out.item():.6f}, expected {ref.item():.6f}; is logits[:, t] scoring ids[:, t+1], masked by the target position?'
""",
        },
        {
            "name": "Packed samples never predict across a boundary",
            "code": """
import torch

def _segment_ref(logits, ids, am, sid):
    # Independent reference: cut every row into its packed samples and score each one alone
    B, T, V = logits.shape
    total, count, sample_means = 0.0, 0, []
    for b in range(B):
        for s in sorted(set(sid[b].tolist()) - {-1}):
            idx = (sid[b] == s).nonzero()[:, 0]
            seg_logits, seg_ids, seg_am = logits[b, idx].double(), ids[b, idx], am[b, idx]
            labels = torch.where(seg_am[1:].bool(), seg_ids[1:], torch.full_like(seg_ids[1:], -100))
            n = int((labels != -100).sum())
            if n == 0:
                continue
            nll = torch.nn.functional.cross_entropy(seg_logits[:-1], labels, ignore_index=-100, reduction='sum')
            total += nll.item()
            count += n
            sample_means.append(nll.item() / n)
    return total / count, sum(sample_means) / len(sample_means)

torch.manual_seed(2)
B, T, V = 2, 12, 9
logits = torch.randn(B, T, V)
ids = torch.randint(0, V, (B, T))
# Continued pre-training on packed rows: every real token is a target
am = torch.ones(B, T, dtype=torch.long)
sid = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1]])
ref_token, _ = _segment_ref(logits, ids, am, sid)
out = {fn}(logits, ids, am, sid, reduction='token')
assert abs(out.item() - ref_token) < 1e-5, f'got {out.item():.6f}, expected {ref_token:.6f}; the last token of one sample must not predict the first token of the next'
""",
        },
        {
            "name": "Sample reduction averages per (row, seq_id)",
            "code": """
import torch

def _segment_ref(logits, ids, am, sid):
    # Independent reference: cut every row into its packed samples and score each one alone
    B, T, V = logits.shape
    total, count, sample_means = 0.0, 0, []
    for b in range(B):
        for s in sorted(set(sid[b].tolist()) - {-1}):
            idx = (sid[b] == s).nonzero()[:, 0]
            seg_logits, seg_ids, seg_am = logits[b, idx].double(), ids[b, idx], am[b, idx]
            labels = torch.where(seg_am[1:].bool(), seg_ids[1:], torch.full_like(seg_ids[1:], -100))
            n = int((labels != -100).sum())
            if n == 0:
                continue
            nll = torch.nn.functional.cross_entropy(seg_logits[:-1], labels, ignore_index=-100, reduction='sum')
            total += nll.item()
            count += n
            sample_means.append(nll.item() / n)
    return total / count, sum(sample_means) / len(sample_means)

torch.manual_seed(3)
B, T, V = 2, 14, 10
logits = torch.randn(B, T, V) * 2
ids = torch.randint(0, V, (B, T))
sid = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, -1, -1]])
am = torch.zeros(B, T, dtype=torch.long)
am[0, 2:9] = 1     # long answer in sample (0, 0)
am[0, 12:14] = 1   # short answer in sample (0, 1)
am[1, 2:3] = 1     # one-token answer in sample (1, 0)
am[1, 5:9] = 1     # sample (1, 1)
# sample (1, 2) is user-only: no counted target, so it must not enter the sample mean
ref_token, ref_sample = _segment_ref(logits, ids, am, sid)
out_token = {fn}(logits, ids, am, sid, reduction='token')
out_sample = {fn}(logits, ids, am, sid, reduction='sample')
assert abs(out_token.item() - ref_token) < 1e-5, f'token: got {out_token.item():.6f}, expected {ref_token:.6f}'
assert abs(out_sample.item() - ref_sample) < 1e-5, f'sample: got {out_sample.item():.6f}, expected {ref_sample:.6f}; each (row, seq_id) is its own sample, and samples with no target are skipped'
assert abs(ref_token - ref_sample) > 1e-3, 'test setup: the two reductions should differ here'
""",
        },
        {
            "name": "Padding is never a target",
            "code": """
import torch

def _segment_ref(logits, ids, am, sid):
    # Independent reference: cut every row into its packed samples and score each one alone
    B, T, V = logits.shape
    total, count, sample_means = 0.0, 0, []
    for b in range(B):
        for s in sorted(set(sid[b].tolist()) - {-1}):
            idx = (sid[b] == s).nonzero()[:, 0]
            seg_logits, seg_ids, seg_am = logits[b, idx].double(), ids[b, idx], am[b, idx]
            labels = torch.where(seg_am[1:].bool(), seg_ids[1:], torch.full_like(seg_ids[1:], -100))
            n = int((labels != -100).sum())
            if n == 0:
                continue
            nll = torch.nn.functional.cross_entropy(seg_logits[:-1], labels, ignore_index=-100, reduction='sum')
            total += nll.item()
            count += n
            sample_means.append(nll.item() / n)
    return total / count, sum(sample_means) / len(sample_means)

torch.manual_seed(4)
B, T, V = 2, 10, 8
logits = torch.randn(B, T, V)
ids = torch.randint(0, V, (B, T))
# assistant_mask is (wrongly) set on padding too; seq_ids == -1 must still exclude it
am = torch.ones(B, T, dtype=torch.long)
am[:, :2] = 0
sid = torch.tensor([[0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1, -1]])
ref_token, ref_sample = _segment_ref(logits, ids, am, sid)
for red, ref in [('token', ref_token), ('sample', ref_sample)]:
    out = {fn}(logits, ids, am, sid, reduction=red)
    assert abs(out.item() - ref) < 1e-5, f'{red}: got {out.item():.6f}, expected {ref:.6f}; padding (seq_id -1) leaked into the loss'
""",
        },
        {
            "name": "No trainable target returns 0 and still backprops",
            "code": """
import torch
torch.manual_seed(5)
B, T, V = 2, 6, 7
logits = torch.randn(B, T, V, requires_grad=True)
ids = torch.randint(0, V, (B, T))
am = torch.zeros(B, T, dtype=torch.long)
am[0, 0] = 1  # position 0 is never a target
sid = torch.zeros(B, T, dtype=torch.long)
for red in ['token', 'sample']:
    out = {fn}(logits, ids, am, sid, reduction=red)
    assert out.dim() == 0, f'{red}: expected a scalar, got shape {tuple(out.shape)}'
    assert out.item() == 0.0, f'{red}: expected 0.0, got {out.item()}'
    out.backward()
assert logits.grad is not None and logits.grad.abs().sum() == 0, 'Gradient should exist and be all zeros'
""",
        },
        {
            "name": "Gradients match the reference and inputs are not mutated",
            "code": """
import torch
torch.manual_seed(6)
B, T, V = 2, 9, 12
base = torch.randn(B, T, V)
ids = torch.randint(0, V, (B, T))
am = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1, 0, 0]])
sid = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -1, -1]])
i0, a0, s0 = ids.clone(), am.clone(), sid.clone()
logits = base.clone().requires_grad_(True)
{fn}(logits, ids, am, sid).backward()
ref_logits = base.clone().requires_grad_(True)
labels = torch.where(am.bool() & (sid >= 0), ids, torch.full_like(ids, -100))
torch.nn.functional.cross_entropy(ref_logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100).backward()
assert logits.grad is not None, 'No gradient reached logits'
assert torch.allclose(logits.grad, ref_logits.grad, atol=1e-6), f'Gradient max diff {(logits.grad - ref_logits.grad).abs().max().item():.2e}'
assert torch.equal(ids, i0) and torch.equal(am, a0) and torch.equal(sid, s0), 'An input tensor was modified in place'
""",
        },
        {
            "name": "Unknown reduction raises ValueError",
            "code": """
import torch
logits = torch.randn(1, 4, 5)
ids = torch.randint(0, 5, (1, 4))
am = torch.ones(1, 4, dtype=torch.long)
sid = torch.zeros(1, 4, dtype=torch.long)
try:
    {fn}(logits, ids, am, sid, reduction='mean')
except ValueError:
    pass
else:
    raise AssertionError('reduction="mean" should raise ValueError')
""",
        },
    ],
    "solution": '''def sft_loss(logits, input_ids, assistant_mask, seq_ids, reduction="token"):
    if reduction not in ("token", "sample"):
        raise ValueError(f"unknown reduction: {reduction}")
    logits = logits[:, :-1]                                  # position t predicts token t+1
    targets = input_ids[:, 1:]
    nll = torch.logsumexp(logits, dim=-1) - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    cur, nxt = seq_ids[:, :-1], seq_ids[:, 1:]
    # Target is an assistant token, in the same packed sample, and not padding
    valid = assistant_mask[:, 1:].bool() & (nxt == cur) & (nxt != -1)
    if not valid.any():
        return logits.sum() * 0.0                            # keeps the graph alive
    if reduction == "token":
        return nll[valid].sum() / valid.sum()
    # One key per (row, seq_id): the same seq_id in another row is another sample
    rows = torch.arange(nll.shape[0], device=nll.device).unsqueeze(1).expand_as(nxt)
    keys = rows[valid] * (int(seq_ids.max()) + 1) + nxt[valid]
    uniq, inv = torch.unique(keys, return_inverse=True)
    sums = nll.new_zeros(len(uniq)).index_add(0, inv, nll[valid])
    counts = nll.new_zeros(len(uniq)).index_add(0, inv, torch.ones_like(nll[valid]))
    return (sums / counts).mean()''',
    "demo": """import torch

torch.manual_seed(0)
V = 20
# One row packing two chats: [user, assistant] [user, assistant], then padding
input_ids = torch.randint(0, V, (1, 12))
assistant_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]])
seq_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1]])
logits = torch.randn(1, 12, V)

print('token-mean :', round(sft_loss(logits, input_ids, assistant_mask, seq_ids).item(), 4))
print('sample-mean:', round(sft_loss(logits, input_ids, assistant_mask, seq_ids, reduction='sample').item(), 4))""",

}
