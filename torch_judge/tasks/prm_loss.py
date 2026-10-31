"""Process Reward Model Loss task."""

TASK = {
    "title": "Process Reward Model Loss",
    "title_zh": "过程奖励模型损失",
    "difficulty": "Medium",
    "description_en": "Train a **process reward model** (PRM): instead of one score for a whole solution, it scores every reasoning step. The solution is split into steps, a special step tag token is placed at the end of each step, and a scalar head on top of the language model reads its output at each tag position as the logit of \"this step is correct\".\n\n**Signature:** `prm_loss(step_logits, input_ids, step_tag_id, step_labels, agg=\"min\") -> tuple[Tensor, Tensor]`\n\n**Parameters:**\n- `step_logits` — scalar head output at every position, shape (B, T)\n- `input_ids` — token ids (B, T), int64; positions where `input_ids == step_tag_id` are step ends. Row $i$ contains $S_i \\ge 1$ tags.\n- `step_tag_id` — id of the step tag token\n- `step_labels` — (B, $S_{\\max}$) float, 1 = correct step, 0 = wrong step, padded with $-1$. Row $i$ holds exactly $S_i$ real labels, in the same order as its tags (the $k$-th tag of the row gets the $k$-th label).\n- `agg` — how step scores combine into a solution score: `\"min\"`, `\"prod\"` or `\"last\"`\n\n**Returns:** `(loss, scores)`\n- `loss` — scalar: binary cross-entropy with logits, averaged over **all real steps in the batch** (every step weighs the same, regardless of which row it is in)\n- `scores` — (B,) solution scores, no gradient. With $p_{i,k} = \\sigma(\\ell_{i,k})$ the probability of the $k$-th step of row $i$:\n$$\n\\text{min}: \\min_k p_{i,k} \\qquad \\text{prod}: \\prod_k p_{i,k} \\qquad \\text{last}: p_{i, S_i}\n$$\n\n**Constraints:**\n- Only the logits at tag positions enter the loss; the gradient at every other position is exactly 0.\n- The loss must stay finite and accurate for large logits ($|\\ell| \\approx 100$): use the log-sigmoid / BCE-with-logits form, not `log(sigmoid(...))`.\n- Unknown `agg` → `ValueError`.\n- Do not modify the inputs.",
    "description_zh": "训练**过程奖励模型**（Process Reward Model，PRM）：它不是给整个解答打一个分，而是给每一步推理打分。解答被切成若干步，每一步末尾放一个特殊的步骤标记 token；语言模型上的标量头在每个标记位置的输出，就是「这一步正确」的 logit。\n\n**签名:** `prm_loss(step_logits, input_ids, step_tag_id, step_labels, agg=\"min\") -> tuple[Tensor, Tensor]`\n\n**参数:**\n- `step_logits` — 标量头在每个位置的输出，形状 (B, T)\n- `input_ids` — token id (B, T)，int64；`input_ids == step_tag_id` 的位置是步骤结尾。第 $i$ 行有 $S_i \\ge 1$ 个标记。\n- `step_tag_id` — 步骤标记 token 的 id\n- `step_labels` — (B, $S_{\\max}$) 浮点，1 表示该步正确，0 表示错误，用 $-1$ 填充。第 $i$ 行恰好有 $S_i$ 个真实标签，顺序与该行的标记一致（第 $k$ 个标记对应第 $k$ 个标签）。\n- `agg` — 各步分数如何合成整个解答的分数：`\"min\"`、`\"prod\"` 或 `\"last\"`\n\n**返回:** `(loss, scores)`\n- `loss` — 标量：带 logits 的二元交叉熵，在**整个 batch 的所有真实步骤**上取平均（每一步权重相同，与它在哪一行无关）\n- `scores` — (B,) 每个解答的分数，不带梯度。记 $p_{i,k} = \\sigma(\\ell_{i,k})$ 为第 $i$ 行第 $k$ 步的概率：\n$$\n\\text{min}: \\min_k p_{i,k} \\qquad \\text{prod}: \\prod_k p_{i,k} \\qquad \\text{last}: p_{i, S_i}\n$$\n\n**约束:**\n- 只有标记位置的 logits 进入损失；其他位置的梯度必须恰好为 0。\n- logits 很大（$|\\ell| \\approx 100$）时损失也要有限且准确：用 log-sigmoid / 带 logits 的 BCE 形式，不要写 `log(sigmoid(...))`。\n- 未知的 `agg` → `ValueError`。\n- 不得修改输入。",
    "function_name": "prm_loss",
    "hint": "is_tag = input_ids == step_tag_id                 # (B, T)\nlogits = step_logits[is_tag]                      # all steps, row by row, in order\nlabels = step_labels[step_labels >= 0]            # same order\nloss = F.binary_cross_entropy_with_logits(logits, labels.float())\nper row: p = sigmoid(step_logits[i][is_tag[i]]); min / prod / last\nscores = stack(...).detach()",
    "hint_zh": "is_tag = input_ids == step_tag_id                 # (B, T)\nlogits = step_logits[is_tag]                      # 所有步骤，按行、按顺序排列\nlabels = step_labels[step_labels >= 0]            # 顺序相同\nloss = F.binary_cross_entropy_with_logits(logits, labels.float())\n每一行：p = sigmoid(step_logits[i][is_tag[i]])；取 min / prod / last\nscores = stack(...).detach()",
    "tests": [
        {
            "name": "Returns a scalar loss and per-row scores without gradient",
            "code": """
import torch
def make_batch(seed, B, T, tag, steps_per_row, V=20):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    ids[ids == tag] = tag + 1  # no accidental tags
    S = max(steps_per_row)
    labels = torch.full((B, S), -1.0)
    for i, s in enumerate(steps_per_row):
        pos = torch.randperm(T, generator=g)[:s].sort().values
        ids[i, pos] = tag
        labels[i, :s] = torch.randint(0, 2, (s,), generator=g).float()
    logits = torch.randn(B, T, generator=g) * 2
    return logits, ids, labels

def ref_steps(logits, ids, tag):
    return [logits[i][ids[i] == tag] for i in range(ids.shape[0])]

logits, ids, labels = make_batch(0, 3, 10, 7, [2, 3, 1])
logits.requires_grad_(True)
loss, scores = {fn}(logits, ids, 7, labels)
assert loss.dim() == 0, f'loss should be a scalar, got shape {tuple(loss.shape)}'
assert loss.requires_grad, 'loss should carry a gradient'
assert scores.shape == (3,), f'scores should have shape (3,), got {tuple(scores.shape)}'
assert not scores.requires_grad, 'scores should be detached'
assert torch.isfinite(loss), 'loss should be finite'
""",
        },
        {
            "name": "Loss averages BCE over every real step in the batch",
            "code": """
import torch
def make_batch(seed, B, T, tag, steps_per_row, V=20):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    ids[ids == tag] = tag + 1  # no accidental tags
    S = max(steps_per_row)
    labels = torch.full((B, S), -1.0)
    for i, s in enumerate(steps_per_row):
        pos = torch.randperm(T, generator=g)[:s].sort().values
        ids[i, pos] = tag
        labels[i, :s] = torch.randint(0, 2, (s,), generator=g).float()
    logits = torch.randn(B, T, generator=g) * 2
    return logits, ids, labels

def ref_steps(logits, ids, tag):
    return [logits[i][ids[i] == tag] for i in range(ids.shape[0])]

# Rows have 1, 4 and 2 steps, so a per-row mean would weigh steps unequally.
# Tags sit at position 0 and at T-1 too.
g = torch.Generator().manual_seed(1)
ids = torch.randint(0, 5, (3, 12), generator=g)   # ids 0..4, tag is 5
ids[0, 0] = 5
ids[1, [2, 5, 8, 11]] = 5
ids[2, [3, 11]] = 5
labels = torch.tensor([[1.0, -1.0, -1.0, -1.0],
                       [1.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, -1.0, -1.0]])
logits = torch.randn(3, 12, generator=g) * 2
logs = torch.cat(ref_steps(logits, ids, 5))
labs = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
p = torch.sigmoid(logs.double())
ref = -(labs * torch.log(p) + (1 - labs) * torch.log(1 - p)).mean()
loss, _ = {fn}(logits, ids, 5, labels)
assert torch.allclose(loss.double(), ref, atol=1e-5), f'loss: got {loss.item():.6f}, expected {ref.item():.6f}'
""",
        },
        {
            "name": "Scores aggregate step probabilities with min, prod and last",
            "code": """
import torch
def make_batch(seed, B, T, tag, steps_per_row, V=20):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    ids[ids == tag] = tag + 1  # no accidental tags
    S = max(steps_per_row)
    labels = torch.full((B, S), -1.0)
    for i, s in enumerate(steps_per_row):
        pos = torch.randperm(T, generator=g)[:s].sort().values
        ids[i, pos] = tag
        labels[i, :s] = torch.randint(0, 2, (s,), generator=g).float()
    logits = torch.randn(B, T, generator=g) * 2
    return logits, ids, labels

def ref_steps(logits, ids, tag):
    return [logits[i][ids[i] == tag] for i in range(ids.shape[0])]

logits, ids, labels = make_batch(2, 4, 9, 3, [3, 1, 4, 2])
steps = ref_steps(logits, ids, 3)
exp = {
    'min': torch.stack([torch.sigmoid(s).min() for s in steps]),
    'prod': torch.stack([torch.sigmoid(s).prod() for s in steps]),
    'last': torch.stack([torch.sigmoid(s)[-1] for s in steps]),
}
for agg, e in exp.items():
    _, scores = {fn}(logits, ids, 3, labels, agg=agg)
    assert torch.allclose(scores, e, atol=1e-6), f'agg={agg}: got {scores.tolist()}, expected {e.tolist()}'
# The default aggregation is min
_, scores = {fn}(logits, ids, 3, labels)
assert torch.allclose(scores, exp['min'], atol=1e-6), f'default agg should be min: got {scores.tolist()}, expected {exp["min"].tolist()}'
""",
        },
        {
            "name": "Gradient reaches only the tag positions",
            "code": """
import torch
def make_batch(seed, B, T, tag, steps_per_row, V=20):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    ids[ids == tag] = tag + 1  # no accidental tags
    S = max(steps_per_row)
    labels = torch.full((B, S), -1.0)
    for i, s in enumerate(steps_per_row):
        pos = torch.randperm(T, generator=g)[:s].sort().values
        ids[i, pos] = tag
        labels[i, :s] = torch.randint(0, 2, (s,), generator=g).float()
    logits = torch.randn(B, T, generator=g) * 2
    return logits, ids, labels

def ref_steps(logits, ids, tag):
    return [logits[i][ids[i] == tag] for i in range(ids.shape[0])]

logits, ids, labels = make_batch(3, 2, 11, 9, [3, 2])
x = logits.clone().requires_grad_(True)
loss, _ = {fn}(x, ids, 9, labels)
loss.backward()
is_tag = ids == 9
n = int(is_tag.sum())
assert x.grad is not None, 'no gradient reached step_logits'
assert torch.all(x.grad[~is_tag] == 0), 'positions without a step tag must get exactly zero gradient'
expected = torch.zeros_like(logits)
for i in range(2):
    pos = is_tag[i].nonzero().flatten()
    expected[i, pos] = (torch.sigmoid(logits[i, pos]) - labels[i, :len(pos)]) / n
assert torch.allclose(x.grad, expected, atol=1e-6), f'gradient at tag positions should be (sigmoid - label) / num_steps; max diff {(x.grad - expected).abs().max().item():.2e}'
""",
        },
        {
            "name": "Stable for large logits",
            "code": """
import torch
ids = torch.tensor([[1, 4, 2, 4, 3, 4]])
labels = torch.tensor([[1.0, 0.0, 0.0]])
# Confident and right, confident and wrong, confident and right
logits = torch.tensor([[0.0, 100.0, 0.0, 100.0, 0.0, -100.0]])
loss, scores = {fn}(logits, ids, 4, labels)
# Per-step BCE: ~0, 100, ~0  ->  mean = 100/3
assert torch.isfinite(loss), f'loss must stay finite for large logits, got {loss.item()}'
assert abs(loss.item() - 100.0 / 3) < 1e-3, f'loss: got {loss.item():.4f}, expected {100.0 / 3:.4f}'
assert torch.isfinite(scores).all(), 'scores must be finite'
""",
        },
        {
            "name": "Unknown aggregation raises ValueError",
            "code": """
import torch
def make_batch(seed, B, T, tag, steps_per_row, V=20):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    ids[ids == tag] = tag + 1  # no accidental tags
    S = max(steps_per_row)
    labels = torch.full((B, S), -1.0)
    for i, s in enumerate(steps_per_row):
        pos = torch.randperm(T, generator=g)[:s].sort().values
        ids[i, pos] = tag
        labels[i, :s] = torch.randint(0, 2, (s,), generator=g).float()
    logits = torch.randn(B, T, generator=g) * 2
    return logits, ids, labels

def ref_steps(logits, ids, tag):
    return [logits[i][ids[i] == tag] for i in range(ids.shape[0])]

logits, ids, labels = make_batch(4, 2, 6, 2, [1, 2])
try:
    {fn}(logits, ids, 2, labels, agg='mean')
except ValueError:
    pass
else:
    raise AssertionError("agg='mean' should raise ValueError")
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
def make_batch(seed, B, T, tag, steps_per_row, V=20):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, T), generator=g)
    ids[ids == tag] = tag + 1  # no accidental tags
    S = max(steps_per_row)
    labels = torch.full((B, S), -1.0)
    for i, s in enumerate(steps_per_row):
        pos = torch.randperm(T, generator=g)[:s].sort().values
        ids[i, pos] = tag
        labels[i, :s] = torch.randint(0, 2, (s,), generator=g).float()
    logits = torch.randn(B, T, generator=g) * 2
    return logits, ids, labels

def ref_steps(logits, ids, tag):
    return [logits[i][ids[i] == tag] for i in range(ids.shape[0])]

logits, ids, labels = make_batch(5, 3, 8, 4, [2, 2, 3])
l0, i0, b0 = logits.clone(), ids.clone(), labels.clone()
{fn}(logits, ids, 4, labels, agg='prod')
assert torch.equal(logits, l0), 'step_logits was modified in place'
assert torch.equal(ids, i0), 'input_ids was modified in place'
assert torch.equal(labels, b0), 'step_labels was modified in place'
""",
        },
    ],
    "solution": '''def prm_loss(step_logits, input_ids, step_tag_id, step_labels, agg="min"):
    if agg not in ("min", "prod", "last"):
        raise ValueError(f"unknown agg: {agg}")
    is_tag = input_ids == step_tag_id                  # (B, T)
    # Boolean indexing walks row by row, left to right, so the k-th selected
    # logit of a row lines up with that row's k-th real label
    logits = step_logits[is_tag]                       # (num_steps,)
    labels = step_labels[step_labels >= 0].to(logits.dtype)
    # BCE with logits uses log-sigmoid internally: finite for any logit
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    with torch.no_grad():
        scores = []
        for i in range(step_logits.shape[0]):
            p = torch.sigmoid(step_logits[i][is_tag[i]])
            if agg == "min":
                scores.append(p.min())
            elif agg == "prod":
                scores.append(p.prod())
            else:
                scores.append(p[-1])
        scores = torch.stack(scores)
    return loss, scores''',
    "demo": """import torch

torch.manual_seed(0)
TAG = 99
# Two solutions: 3 steps and 2 steps; the tag token closes each step
ids = torch.tensor([[5, 6, TAG, 7, 8, TAG, 9, TAG],
                    [4, TAG, 3, 2, 1, TAG, 0, 0]])
labels = torch.tensor([[1.0, 1.0, 0.0],
                       [1.0, 1.0, -1.0]])
logits = torch.randn(2, 8, requires_grad=True)

loss, scores = prm_loss(logits, ids, TAG, labels)
loss.backward()
print('loss:', round(loss.item(), 4))
for agg in ('min', 'prod', 'last'):
    print(f'{agg:>4} scores:', prm_loss(logits, ids, TAG, labels, agg=agg)[1].tolist())
print('grad is zero off the tags:', bool((logits.grad[ids != TAG] == 0).all()))""",

}
