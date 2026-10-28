"""Sampling Penalties and Min-p task."""

TASK = {
    "title": "Sampling Penalties & Min-p",
    "title_zh": "采样惩罚与 Min-p",
    "difficulty": "Medium",
    "description_en": "Implement the logits processors that inference engines such as vLLM and Hugging Face apply before sampling the next token: repetition, frequency and presence penalties, temperature scaling, and min-p filtering.\n\nThe penalties discourage the model from repeating itself, and min-p removes tokens that are unlikely relative to the most likely one. The order of the steps matters and is fixed below.\n\n**Signature:** `apply_sampling_penalties(logits, prev_tokens, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, temperature=1.0, min_p=0.0) -> Tensor`\n\n**Parameters:**\n- `logits` — float tensor `(V,)` of raw logits\n- `prev_tokens` — list of token ids generated so far; may contain repeats, may be empty\n- `repetition_penalty` — CTRL-style multiplicative penalty (1.0 = off)\n- `frequency_penalty` — subtracted once per occurrence of the token\n- `presence_penalty` — subtracted once if the token occurred at all\n- `temperature` — divides all logits; must be > 0\n- `min_p` — in [0, 1]; 0 = off\n\n**Returns:** probability tensor `(V,)` that sums to 1.\n\n**Formula:** apply these steps in exactly this order:\n```\n1. repetition, once per DISTINCT token t in prev_tokens:\n     logit[t] = logit[t] / repetition_penalty   if logit[t] > 0\n     logit[t] = logit[t] * repetition_penalty   otherwise\n2. frequency and presence, count[t] = occurrences of t in prev_tokens:\n     logit[t] -= frequency_penalty * count[t] + presence_penalty * (count[t] > 0)\n3. logits = logits / temperature\n4. probs = softmax(logits)\n5. min-p: set probs[i] = 0 where probs[i] < min_p * max(probs), then probs = probs / sum(probs)\n```\n\n**Constraints:**\n- Tokens that do not appear in `prev_tokens` are left unchanged by steps 1 and 2\n- Penalties are applied to the logits before temperature (step 3), not after\n- A token whose probability equals the min-p threshold is kept (only strictly smaller probabilities are removed)\n- Do not modify the input `logits` tensor or the `prev_tokens` list",
    "description_zh": "实现推理引擎（如 vLLM、Hugging Face）在采样下一个 token 之前使用的 logits 处理器：重复惩罚、频率惩罚与存在惩罚、温度缩放，以及 min-p 过滤。\n\n这些惩罚用于抑制模型重复自身输出，min-p 则剔除相对于最可能 token 而言概率过低的 token。步骤的顺序很重要，且以下面给出的为准。\n\n**签名:** `apply_sampling_penalties(logits, prev_tokens, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, temperature=1.0, min_p=0.0) -> Tensor`\n\n**参数:**\n- `logits` — 形状 `(V,)` 的 float 张量，为原始 logits\n- `prev_tokens` — 目前已生成的 token id 列表；可含重复项，可为空\n- `repetition_penalty` — CTRL 风格的乘性惩罚（1.0 = 关闭）\n- `frequency_penalty` — 按该 token 每次出现各减一次（随次数累加）\n- `presence_penalty` — 只要该 token 出现过，就减一次\n- `temperature` — 所有 logits 都除以该值；必须 > 0\n- `min_p` — 取值范围 [0, 1]；0 = 关闭\n\n**返回:** 形状 `(V,)` 的概率张量，和为 1。\n\n**公式:** 严格按以下顺序执行各步骤：\n```\n1. 重复惩罚，对 prev_tokens 中每个不同的 token t 只执行一次：\n     logit[t] = logit[t] / repetition_penalty   若 logit[t] > 0\n     logit[t] = logit[t] * repetition_penalty   否则\n2. 频率与存在惩罚，count[t] 为 t 在 prev_tokens 中的出现次数：\n     logit[t] -= frequency_penalty * count[t] + presence_penalty * (count[t] > 0)\n3. logits = logits / temperature\n4. probs = softmax(logits)\n5. min-p：当 probs[i] < min_p * max(probs) 时令 probs[i] = 0，然后 probs = probs / sum(probs)\n```\n\n**约束:**\n- 未出现在 `prev_tokens` 中的 token，在步骤 1 与 2 中保持不变\n- 惩罚作用于温度缩放（步骤 3）之前的 logits，而非之后\n- 概率恰好等于 min-p 阈值的 token 保留（仅严格小于阈值的概率被剔除）\n- 不得修改输入的 `logits` 张量与 `prev_tokens` 列表",
    "function_name": "apply_sampling_penalties",
    "hint": "counts = Counter(prev_tokens)\nfor t in counts:  logit[t] = logit[t]/rp if logit[t] > 0 else logit[t]*rp   # once per token\nfor t, c in counts.items():  logit[t] -= fp*c + pp\nprobs = softmax(logit / temperature)\nprobs = where(probs < min_p * probs.max(), 0, probs); probs /= probs.sum()",
    "hint_zh": "counts = Counter(prev_tokens)\nfor t in counts:  logit[t] = logit[t]/rp if logit[t] > 0 else logit[t]*rp   # 每个 token 一次\nfor t, c in counts.items():  logit[t] -= fp*c + pp\nprobs = softmax(logit / temperature)\nprobs = where(probs < min_p * probs.max(), 0, probs); probs /= probs.sum()",
    "tests": [
        {
            "name": "Default settings equal plain softmax",
            "code": """
import torch
torch.manual_seed(0)
logits = torch.randn(50)
out = {fn}(logits, [1, 2, 3])
ref = torch.softmax(logits, dim=-1)
assert out.shape == (50,), f'shape should be (50,), got {tuple(out.shape)}'
assert torch.allclose(out, ref, atol=1e-6), 'default penalties should give plain softmax'
""",
        },
        {
            "name": "Repetition penalty divides positive and multiplies negative logits",
            "code": """
import torch
logits = torch.tensor([2.0, -2.0, 1.0, 0.0])
out = {fn}(logits, [0, 1], repetition_penalty=2.0)
# token 0: 2.0 / 2 = 1.0 ; token 1: -2.0 * 2 = -4.0 ; others unchanged
expected = torch.softmax(torch.tensor([1.0, -4.0, 1.0, 0.0]), dim=-1)
assert torch.allclose(out, expected, atol=1e-6), f'got {out.tolist()}, expected {expected.tolist()}'
""",
        },
        {
            "name": "Repetition penalty applied once per distinct token",
            "code": """
import torch
logits = torch.tensor([1.0, 0.0, 0.0, 0.0])
out = {fn}(logits, [0, 0, 0], repetition_penalty=2.0)
# token 0 occurs 3 times but the repetition penalty applies once: 1.0 / 2 = 0.5
expected = torch.softmax(torch.tensor([0.5, 0.0, 0.0, 0.0]), dim=-1)
assert torch.allclose(out, expected, atol=1e-6), f'got {out.tolist()}, expected {expected.tolist()}'
""",
        },
        {
            "name": "Frequency scales with count, presence applies once",
            "code": """
import torch
logits = torch.zeros(4)
out = {fn}(logits, [2, 2, 2], frequency_penalty=0.5, presence_penalty=1.0)
# token 2: 0 - (0.5 * 3 + 1.0 * 1) = -2.5 ; others unchanged
expected = torch.softmax(torch.tensor([0.0, 0.0, -2.5, 0.0]), dim=-1)
assert torch.allclose(out, expected, atol=1e-6), f'got {out.tolist()}, expected {expected.tolist()}'
""",
        },
        {
            "name": "Penalties are applied before temperature",
            "code": """
import torch
logits = torch.tensor([1.0, 0.0])
out = {fn}(logits, [0], frequency_penalty=1.0, temperature=2.0)
# penalty first: [1 - 1, 0] = [0, 0], then / 2 -> uniform
# temperature first would give [0.5 - 1, 0] = [-0.5, 0], which is not uniform
expected = torch.tensor([0.5, 0.5])
assert torch.allclose(out, expected, atol=1e-6), f'got {out.tolist()}, expected uniform {expected.tolist()}'
""",
        },
        {
            "name": "Min-p removes low-probability tokens and renormalizes",
            "code": """
import torch
probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
logits = torch.log(probs)
out = {fn}(logits, [], min_p=0.2)
# threshold = 0.2 * 0.5 = 0.1 -> drop 0.05, renormalize the rest (sum 0.95)
expected = torch.tensor([0.5, 0.3, 0.15, 0.0]) / 0.95
assert torch.allclose(out, expected, atol=1e-5), f'got {out.tolist()}, expected {expected.tolist()}'
assert abs(out.sum().item() - 1.0) < 1e-6, f'output should sum to 1, got {out.sum().item()}'
out2 = {fn}(logits, [], min_p=0.5)
# threshold = 0.25 -> keep 0.5 and 0.3 only, renormalize (sum 0.8)
expected2 = torch.tensor([0.5, 0.3, 0.0, 0.0]) / 0.8
assert torch.allclose(out2, expected2, atol=1e-5), f'got {out2.tolist()}, expected {expected2.tolist()}'
""",
        },
        {
            "name": "Empty prev_tokens",
            "code": """
import torch
torch.manual_seed(2)
logits = torch.randn(20)
out = {fn}(logits, [], repetition_penalty=1.5, frequency_penalty=1.0, presence_penalty=1.0, temperature=0.7)
expected = torch.softmax(logits / 0.7, dim=-1)
assert torch.allclose(out, expected, atol=1e-6), 'with no history only temperature should apply'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(3)
logits = torch.randn(30)
orig = logits.clone()
prev = [1, 1, 4, 7]
prev_copy = list(prev)
{fn}(logits, prev, repetition_penalty=1.2, frequency_penalty=0.3, presence_penalty=0.1, temperature=0.9, min_p=0.05)
assert torch.equal(logits, orig), 'logits tensor was modified in place'
assert prev == prev_copy, 'prev_tokens list was modified'
""",
        },
        {
            "name": "Matches explicit reference and sums to one",
            "code": """
import torch
torch.manual_seed(4)
V = 200
logits = torch.randn(V) * 2
prev = torch.randint(0, V, (60,)).tolist()
rp, fp, pp, temp, mp = 1.3, 0.1, 0.2, 0.8, 0.02
out = {fn}(logits, prev, repetition_penalty=rp, frequency_penalty=fp, presence_penalty=pp, temperature=temp, min_p=mp)
# reference with explicit loops
ref = logits.clone().tolist()
counts = {}
for t in prev:
    counts[t] = counts.get(t, 0) + 1
for t in counts:
    ref[t] = ref[t] / rp if ref[t] > 0 else ref[t] * rp
for t, c in counts.items():
    ref[t] = ref[t] - (fp * c + pp * (1 if c > 0 else 0))
p = torch.softmax(torch.tensor(ref, dtype=torch.float64) / temp, dim=-1)
p = torch.where(p < mp * p.max(), torch.zeros_like(p), p)
p = p / p.sum()
assert out.shape == (V,), f'shape should be ({V},), got {tuple(out.shape)}'
assert abs(out.sum().item() - 1.0) < 1e-5, f'output should sum to 1, got {out.sum().item()}'
assert torch.allclose(out.double(), p, atol=1e-5), 'output differs from reference'
""",
        },
    ],
    "solution": '''from collections import Counter


def apply_sampling_penalties(logits, prev_tokens, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, temperature=1.0, min_p=0.0):
    logits = logits.clone()  # never modify the caller's tensor
    counts = Counter(prev_tokens)

    if counts:
        ids = torch.tensor(list(counts.keys()), dtype=torch.long)
        cnt = torch.tensor(list(counts.values()), dtype=logits.dtype)
        vals = logits[ids]

        # 1. Repetition penalty: once per distinct token, sign-aware
        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        # 2. Frequency scales with the count; presence applies once (every id has count > 0)
        vals = vals - (frequency_penalty * cnt + presence_penalty)

        logits[ids] = vals

    # 3. Temperature
    logits = logits / temperature

    # 4. Softmax
    probs = torch.softmax(logits, dim=-1)

    # 5. Min-p: drop tokens far below the most likely one, then renormalize
    threshold = min_p * probs.max()
    probs = torch.where(probs < threshold, torch.zeros_like(probs), probs)
    return probs / probs.sum()''',
    "demo": """torch.manual_seed(0)
V = 8
logits = torch.randn(V)
prev = [3, 3, 5, 1]

base = torch.softmax(logits, dim=-1)
print('plain softmax:  ', [round(x, 3) for x in base.tolist()])

penalized = apply_sampling_penalties(logits, prev, repetition_penalty=1.5, frequency_penalty=0.3, presence_penalty=0.2, temperature=0.8, min_p=0.05)
print('with penalties: ', [round(x, 3) for x in penalized.tolist()])
print('sum:', round(penalized.sum().item(), 6))""",

}
