"""Per-Token Log-Probs task."""

TASK = {
    "title": "Per-Token Log-Probs",
    "title_zh": "逐 token 对数概率",
    "difficulty": "Medium",
    "description_en": "Turn a language model's logits into the per-token log-probabilities of the tokens it actually saw, keeping only the response.\n\nEvery post-training loss starts here: SFT, DPO, PPO and GRPO all need $\\log \\pi(y_t \\mid y_{<t})$ for each response token. Two details trip people up: logits at position $t$ score the **next** token, so logits and labels are offset by one; and prompt and padding tokens must not contribute.\n\n**Signature:** `token_logprobs(logits, input_ids, response_mask, temperature=1.0) -> Tensor`\n\n**Parameters:**\n- `logits` — model output on `input_ids`, shape (B, T, V); `logits[:, t]` predicts `input_ids[:, t+1]`\n- `input_ids` — token ids (B, T), int64: prompt followed by response, then right padding\n- `response_mask` — (B, T), 1 (or True) where `input_ids` is a generated response token, 0 for prompt and padding\n- `temperature` — the sampling temperature; logits are divided by it before the softmax\n\n**Returns:** tensor of shape (B, T−1)\n\n**Formula:**\n$$\n\\text{out}[b, t] =\n\\begin{cases}\n\\log \\operatorname{softmax}\\!\\left(\\dfrac{\\text{logits}[b, t]}{\\tau}\\right)[\\,\\text{input\\_ids}[b, t+1]\\,] & \\text{if } \\text{response\\_mask}[b, t+1] = 1 \\\\\n0 & \\text{otherwise}\n\\end{cases}\n$$\n\n**Constraints:**\n- Numerically stable for large logits (values around $10^3$ must give finite results): use the log-sum-exp form, not `log(softmax(...))`.\n- Masked positions are exactly 0.\n- Gradients must flow to `logits`; inputs must not be modified.",
    "description_zh": "把语言模型的 logits 变成它实际看到的每个 token 的对数概率，只保留回答部分。\n\n所有后训练损失都从这一步开始：SFT、DPO、PPO、GRPO 都需要每个回答 token 的 $\\log \\pi(y_t \\mid y_{<t})$。有两个细节最容易出错：位置 $t$ 的 logits 预测的是**下一个** token，所以 logits 和标签要错开一位；提示词和填充部分不能计入。\n\n**签名:** `token_logprobs(logits, input_ids, response_mask, temperature=1.0) -> Tensor`\n\n**参数:**\n- `logits` — 模型在 `input_ids` 上的输出，形状 (B, T, V)；`logits[:, t]` 预测 `input_ids[:, t+1]`\n- `input_ids` — token id (B, T)，int64：先是提示词，接着是回答，最后是右侧填充\n- `response_mask` — (B, T)，`input_ids` 是生成的回答 token 处为 1（或 True），提示词和填充处为 0\n- `temperature` — 采样温度；softmax 之前先把 logits 除以它\n\n**返回:** 形状为 (B, T−1) 的张量\n\n**公式:**\n$$\n\\text{out}[b, t] =\n\\begin{cases}\n\\log \\operatorname{softmax}\\!\\left(\\dfrac{\\text{logits}[b, t]}{\\tau}\\right)[\\,\\text{input\\_ids}[b, t+1]\\,] & \\text{若 } \\text{response\\_mask}[b, t+1] = 1 \\\\\n0 & \\text{否则}\n\\end{cases}\n$$\n\n**约束:**\n- 对很大的 logits 也要数值稳定（数值在 $10^3$ 量级时结果必须有限）：用 log-sum-exp 的形式，不要写 `log(softmax(...))`。\n- 被掩掉的位置恰好为 0。\n- 梯度必须回传到 `logits`；不得修改输入。",
    "function_name": "token_logprobs",
    "hint": "logits = logits[:, :-1] / temperature          # (B, T-1, V): position t scores token t+1\ntargets = input_ids[:, 1:]                     # (B, T-1)\npicked = logits.gather(-1, targets[..., None])[..., 0]\nlogps = picked - logsumexp(logits, dim=-1)     # stable log-softmax at the target\nreturn logps.masked_fill(~response_mask[:, 1:].bool(), 0.0)  # mask by the TARGET position",
    "hint_zh": "logits = logits[:, :-1] / temperature          # (B, T-1, V)：位置 t 给第 t+1 个 token 打分\ntargets = input_ids[:, 1:]                     # (B, T-1)\npicked = logits.gather(-1, targets[..., None])[..., 0]\nlogps = picked - logsumexp(logits, dim=-1)     # 稳定地取目标处的 log-softmax\nreturn logps.masked_fill(~response_mask[:, 1:].bool(), 0.0)  # 按「目标 token」的位置取掩码",
    "tests": [
        {
            "name": "Shape is (B, T-1) and masked positions are zero",
            "code": """
import torch
torch.manual_seed(0)
B, T, V = 2, 7, 11
logits = torch.randn(B, T, V)
input_ids = torch.randint(0, V, (B, T))
response_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])
out = {fn}(logits, input_ids, response_mask)
assert out.shape == (B, T - 1), f'Expected shape {(B, T - 1)}, got {tuple(out.shape)}'
target_is_response = response_mask[:, 1:].bool()
assert (out[~target_is_response] == 0).all(), 'Positions whose target is a prompt or padding token must be exactly 0'
assert (out[target_is_response] < 0).all(), 'Log-probabilities of response tokens must be negative'
""",
        },
        {
            "name": "Matches log_softmax at the next token",
            "code": """
import torch
torch.manual_seed(1)
B, T, V = 3, 9, 17
logits = torch.randn(B, T, V)
input_ids = torch.randint(0, V, (B, T))
response_mask = torch.zeros(B, T, dtype=torch.bool)
response_mask[0, 2:9] = True
response_mask[1, 5:7] = True
response_mask[2, 4:8] = True
# Independent reference: loop over positions in float64
lp = torch.log_softmax(logits.double(), dim=-1)
ref = torch.zeros(B, T - 1, dtype=torch.float64)
for b in range(B):
    for t in range(T - 1):
        if response_mask[b, t + 1]:
            ref[b, t] = lp[b, t, input_ids[b, t + 1]]
out = {fn}(logits, input_ids, response_mask)
assert torch.allclose(out.double(), ref, atol=1e-5), f'Max diff {(out.double() - ref).abs().max().item():.2e}; is logits[:, t] scoring input_ids[:, t+1]?'
""",
        },
        {
            "name": "Temperature divides the logits before the softmax",
            "code": """
import torch
torch.manual_seed(2)
B, T, V = 2, 6, 9
logits = torch.randn(B, T, V) * 3
input_ids = torch.randint(0, V, (B, T))
response_mask = torch.tensor([[0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 0]])
for tau in [0.6, 1.0, 2.0]:
    lp = torch.log_softmax(logits.double() / tau, dim=-1)
    ref = torch.zeros(B, T - 1, dtype=torch.float64)
    for b in range(B):
        for t in range(T - 1):
            if response_mask[b, t + 1]:
                ref[b, t] = lp[b, t, input_ids[b, t + 1]]
    out = {fn}(logits, input_ids, response_mask, temperature=tau)
    assert torch.allclose(out.double(), ref, atol=1e-5), f'temperature={tau}: max diff {(out.double() - ref).abs().max().item():.2e}'
""",
        },
        {
            "name": "Stable for large logits",
            "code": """
import torch
torch.manual_seed(3)
B, T, V = 2, 8, 50
logits = torch.randn(B, T, V) * 1000
input_ids = torch.randint(0, V, (B, T))
response_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 0, 0]])
out = {fn}(logits, input_ids, response_mask)
assert torch.isfinite(out).all(), 'Got inf/nan: log(softmax(x)) underflows for large logits, use logsumexp'
lp = torch.log_softmax(logits.double(), dim=-1)
ref = torch.zeros(B, T - 1, dtype=torch.float64)
for b in range(B):
    for t in range(T - 1):
        if response_mask[b, t + 1]:
            ref[b, t] = lp[b, t, input_ids[b, t + 1]]
assert torch.allclose(out.double(), ref, atol=1e-2, rtol=1e-5), f'Max diff {(out.double() - ref).abs().max().item():.2e}'
""",
        },
        {
            "name": "Gradients flow to logits",
            "code": """
import torch
torch.manual_seed(4)
B, T, V = 2, 7, 12
base = torch.randn(B, T, V)
input_ids = torch.randint(0, V, (B, T))
response_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 0]])
w = torch.randn(B, T - 1)
logits = base.clone().requires_grad_(True)
out = {fn}(logits, input_ids, response_mask, temperature=0.8)
(out * w).sum().backward()
assert logits.grad is not None, 'No gradient reached logits'
ref_logits = base.clone().requires_grad_(True)
lp = torch.log_softmax(ref_logits[:, :-1] / 0.8, dim=-1)
picked = lp.gather(-1, input_ids[:, 1:, None])[..., 0] * response_mask[:, 1:]
(picked * w).sum().backward()
assert torch.allclose(logits.grad, ref_logits.grad, atol=1e-5), f'Gradient max diff {(logits.grad - ref_logits.grad).abs().max().item():.2e}'
assert logits.grad[:, -1].abs().sum() == 0, 'The last position has no next token and must get no gradient'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(5)
B, T, V = 2, 6, 10
logits = torch.randn(B, T, V)
input_ids = torch.randint(0, V, (B, T))
response_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]])
l0, i0, m0 = logits.clone(), input_ids.clone(), response_mask.clone()
{fn}(logits, input_ids, response_mask, temperature=0.7)
assert torch.equal(logits, l0), 'logits were modified in place'
assert torch.equal(input_ids, i0), 'input_ids were modified in place'
assert torch.equal(response_mask, m0), 'response_mask was modified in place'
""",
        },
    ],
    "solution": '''def token_logprobs(logits, input_ids, response_mask, temperature=1.0):
    # Position t scores token t+1, so drop the last logits and the first ids
    logits = logits[:, :-1] / temperature                    # (B, T-1, V)
    targets = input_ids[:, 1:]                               # (B, T-1)
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # log softmax at the target = logit - logsumexp, stable for large logits
    logps = picked - torch.logsumexp(logits, dim=-1)
    # Keep a position only if its TARGET token belongs to the response
    keep = response_mask[:, 1:].bool()
    return logps.masked_fill(~keep, 0.0)''',
    "demo": """import torch

torch.manual_seed(0)
V = 10
# One sequence: 3 prompt tokens, 4 response tokens, 1 padding token
input_ids = torch.tensor([[2, 7, 1, 4, 4, 9, 0, 0]])
response_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0]])
logits = torch.randn(1, 8, V)

logps = token_logprobs(logits, input_ids, response_mask)
print('Per-token log-probs:', logps.round(decimals=3))
print('Sequence log-prob (sum, as used by DPO):', round(logps.sum().item(), 4))""",

}
