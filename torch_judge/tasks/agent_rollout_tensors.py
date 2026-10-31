"""Multi-Turn Agent Loss Mask task."""

TASK = {
    "title": "Multi-Turn Agent Loss Mask",
    "title_zh": "Agent 多轮轨迹掩码",
    "difficulty": "Easy",
    "description_en": "Turn a multi-turn agent trajectory into the tensors an RL trainer consumes, with a loss mask that trains only on what the policy itself generated.\n\nIn agentic RL the model writes an assistant turn, calls a tool, receives the tool's output, writes the next turn, and so on. The whole trajectory is fed to the model as one sequence, but only the **assistant** tokens were sampled from the policy. System prompts, user messages and tool outputs come from the environment: they are context, not actions, so they must not receive policy-gradient updates. A trajectory that ran past the length limit was cut off before it finished, so its reward is unreliable and it is excluded from the loss entirely (overlong filtering).\n\n**Signature:** `agent_rollout_tensors(turns, max_len, pad_id) -> tuple[Tensor, Tensor, Tensor]`\n\n**Parameters:**\n- `turns` — list of `(role, token_ids)` in order; `role` is one of `\"system\"`, `\"user\"`, `\"assistant\"`, `\"tool\"`; `token_ids` is a list of ints (may be empty)\n- `max_len` — fixed output length (positive int)\n- `pad_id` — token id used for padding (it may also occur as a real token)\n\n**Returns:** `(input_ids, attention_mask, loss_mask)`, each an int64 tensor of shape `(max_len,)`\n\n**Rules:**\n- Concatenate the token ids of all turns in order. `loss_mask` is 1 on tokens from `\"assistant\"` turns and 0 on every other role.\n- If the total length $N$ exceeds `max_len`, keep the **first** `max_len` tokens and set the whole `loss_mask` to 0 (the trajectory was truncated). $N = \\texttt{max\\_len}$ is not truncated.\n- Otherwise right-pad to `max_len` with `pad_id`. `attention_mask` is 1 on the real tokens and 0 on padding; `loss_mask` is 0 on padding.\n- An unknown role raises `ValueError`. The input list and its token lists must not be modified.",
    "description_zh": "把一条多轮 Agent 轨迹转成 RL 训练框架要用的张量，并构造一个只在策略自己生成的 token 上计算损失的掩码。\n\n在 Agent RL 里，模型先写一轮助手回答、调用工具、拿到工具输出，再写下一轮，如此往复。整条轨迹作为一个序列喂给模型，但只有**助手**的 token 是从策略里采样出来的。系统提示词、用户消息和工具输出来自环境：它们是上下文而不是动作，不能拿去做策略梯度更新。超过长度上限的轨迹在结束之前就被截断了，奖励不可靠，整条轨迹都不计入损失（超长过滤，overlong filtering）。\n\n**签名:** `agent_rollout_tensors(turns, max_len, pad_id) -> tuple[Tensor, Tensor, Tensor]`\n\n**参数:**\n- `turns` — 按顺序排列的 `(role, token_ids)` 列表；`role` 取 `\"system\"`、`\"user\"`、`\"assistant\"`、`\"tool\"` 之一；`token_ids` 是 int 列表（可以为空）\n- `max_len` — 固定的输出长度（正整数）\n- `pad_id` — 用于填充的 token id（它也可能作为真实 token 出现）\n\n**返回:** `(input_ids, attention_mask, loss_mask)`，每个都是形状为 `(max_len,)` 的 int64 张量\n\n**规则:**\n- 按顺序拼接所有轮次的 token id。`loss_mask` 在 `\"assistant\"` 轮次的 token 上为 1，其他角色都为 0。\n- 如果总长度 $N$ 超过 `max_len`，保留**前** `max_len` 个 token，并把整个 `loss_mask` 置 0（轨迹被截断了）。$N = \\texttt{max\\_len}$ 不算截断。\n- 否则用 `pad_id` 在右侧填充到 `max_len`。`attention_mask` 在真实 token 上为 1、填充处为 0；`loss_mask` 在填充处为 0。\n- 未知的角色抛出 `ValueError`。不得修改输入列表及其中的 token 列表。",
    "function_name": "agent_rollout_tensors",
    "hint": "ids, loss = [], []\nfor role, toks in turns:\n    check role; ids += toks; loss += [1 if role == \"assistant\" else 0] * len(toks)\nn = len(ids)\nif n > max_len: ids, loss = ids[:max_len], [0] * max_len        # truncated: no loss at all\nattn = [1] * len(ids) + [0] * (max_len - len(ids))\npad ids with pad_id and loss with 0 up to max_len\nreturn three torch.long tensors",
    "hint_zh": "ids, loss = [], []\nfor role, toks in turns:\n    检查 role；ids += toks；loss += [1 if role == \"assistant\" else 0] * len(toks)\nn = len(ids)\nif n > max_len: ids, loss = ids[:max_len], [0] * max_len        # 被截断：整条不计损失\nattn = [1] * len(ids) + [0] * (max_len - len(ids))\nids 用 pad_id、loss 用 0 填充到 max_len\n返回三个 torch.long 张量",
    "tests": [
        {
            "name": "Shapes, dtypes and a single exchange",
            "code": """
import torch
turns = [('system', [1, 2]), ('user', [3, 4, 5]), ('assistant', [6, 7, 8])]
ids, attn, loss = {fn}(turns, 10, 0)
for name, x in [('input_ids', ids), ('attention_mask', attn), ('loss_mask', loss)]:
    assert isinstance(x, torch.Tensor) and x.dtype == torch.long, f'{name} must be an int64 tensor'
    assert tuple(x.shape) == (10,), f'{name} must have shape (max_len,), got {tuple(x.shape)}'
assert ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 0, 0], f'input_ids: got {ids.tolist()}'
assert attn.tolist() == [1, 1, 1, 1, 1, 1, 1, 1, 0, 0], f'attention_mask: got {attn.tolist()}'
assert loss.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0], f'loss_mask: got {loss.tolist()}'
""",
        },
        {
            "name": "Tool outputs and user turns are context, not actions",
            "code": """
import torch
turns = [
    ('system', [10]),
    ('user', [11, 12]),
    ('assistant', [20, 21, 22]),   # tool call
    ('tool', [30, 31, 32, 33]),    # tool output from the environment
    ('assistant', [23, 24]),       # answer
    ('user', [13]),
    ('assistant', [25]),
]
ids, attn, loss = {fn}(turns, 16, 99)
expected_loss = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0]
assert ids.tolist()[:14] == [10, 11, 12, 20, 21, 22, 30, 31, 32, 33, 23, 24, 13, 25], f'input_ids: got {ids.tolist()}'
assert loss.tolist() == expected_loss, f'loss_mask must be 1 only on assistant tokens: got {loss.tolist()}, expected {expected_loss}'
""",
        },
        {
            "name": "Padding uses pad_id, and attention comes from length, not from pad_id",
            "code": """
import torch
# pad_id = 5 also appears as a real token inside the trajectory
turns = [('user', [5, 6]), ('assistant', [7, 5, 5])]
ids, attn, loss = {fn}(turns, 8, 5)
assert ids.tolist() == [5, 6, 7, 5, 5, 5, 5, 5], f'input_ids: got {ids.tolist()}'
assert attn.tolist() == [1, 1, 1, 1, 1, 0, 0, 0], f'attention_mask must cover the 5 real tokens even where they equal pad_id: got {attn.tolist()}'
assert loss.tolist() == [0, 0, 1, 1, 1, 0, 0, 0], f'loss_mask: got {loss.tolist()}'
# A different pad_id on the same trajectory
ids2, attn2, _ = {fn}(turns, 7, 42)
assert ids2.tolist() == [5, 6, 7, 5, 5, 42, 42], f'pad with pad_id=42: got {ids2.tolist()}'
assert attn2.tolist() == [1, 1, 1, 1, 1, 0, 0], f'attention_mask: got {attn2.tolist()}'
""",
        },
        {
            "name": "Truncated trajectories keep the first max_len tokens and get no loss",
            "code": """
import torch
turns = [('system', [1, 2]), ('user', [3]), ('assistant', [4, 5, 6]), ('tool', [7, 8]), ('assistant', [9, 10, 11])]
# 11 tokens, max_len 7: truncated
ids, attn, loss = {fn}(turns, 7, 0)
assert ids.tolist() == [1, 2, 3, 4, 5, 6, 7], f'Keep the first max_len tokens (right truncation): got {ids.tolist()}'
assert attn.tolist() == [1] * 7, f'attention_mask: got {attn.tolist()}'
assert loss.tolist() == [0] * 7, f'A truncated trajectory must have an all-zero loss_mask (overlong filtering): got {loss.tolist()}'
""",
        },
        {
            "name": "Exactly max_len tokens is not truncated",
            "code": """
import torch
turns = [('user', [1, 2, 3]), ('assistant', [4, 5]), ('tool', [6]), ('assistant', [7, 8])]
ids, attn, loss = {fn}(turns, 8, 0)
assert ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8], f'input_ids: got {ids.tolist()}'
assert attn.tolist() == [1] * 8, f'attention_mask: got {attn.tolist()}'
assert loss.tolist() == [0, 0, 0, 1, 1, 0, 1, 1], f'N == max_len is not truncated, so the loss mask stays: got {loss.tolist()}'
# One token over the limit is truncated
ids2, attn2, loss2 = {fn}(turns, 7, 0)
assert loss2.tolist() == [0] * 7, f'N = max_len + 1 is truncated: got loss_mask {loss2.tolist()}'
""",
        },
        {
            "name": "Empty turns, empty trajectory and unknown roles",
            "code": """
import torch
ids, attn, loss = {fn}([('system', []), ('user', [3]), ('assistant', []), ('assistant', [4])], 4, 0)
assert ids.tolist() == [3, 4, 0, 0] and attn.tolist() == [1, 1, 0, 0] and loss.tolist() == [0, 1, 0, 0], f'Empty turns contribute nothing: got {ids.tolist()}, {attn.tolist()}, {loss.tolist()}'
ids, attn, loss = {fn}([], 3, 9)
assert ids.tolist() == [9, 9, 9] and attn.tolist() == [0, 0, 0] and loss.tolist() == [0, 0, 0], 'An empty trajectory is all padding'
for bad in ['observation', 'Assistant']:
    try:
        {fn}([('user', [1]), (bad, [2])], 4, 0)
    except ValueError:
        continue
    raise AssertionError(f'Expected ValueError for role {bad!r}')
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import copy
turns = [('user', [1, 2]), ('assistant', [3, 4, 5]), ('tool', [6]), ('assistant', [7])]
before = copy.deepcopy(turns)
{fn}(turns, 5, 0)
{fn}(turns, 12, 0)
assert turns == before, f'The turns list or its token lists were modified: {turns}'
""",
        },
    ],
    "solution": '''def agent_rollout_tensors(turns, max_len, pad_id):
    roles = ("system", "user", "assistant", "tool")
    ids, loss = [], []
    for role, token_ids in turns:
        if role not in roles:
            raise ValueError(f"unknown role: {role}")
        ids += token_ids                                  # += on a new list: inputs stay intact
        # Only tokens the policy generated are trained on
        loss += [1 if role == "assistant" else 0] * len(token_ids)
    if len(ids) > max_len:
        # Truncated before it finished: the reward is unreliable, so no loss at all
        ids, loss = ids[:max_len], [0] * max_len
    n_pad = max_len - len(ids)
    attn = [1] * len(ids) + [0] * n_pad                 # from the length, not from pad_id
    ids = ids + [pad_id] * n_pad
    loss = loss + [0] * n_pad
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long),
            torch.tensor(loss, dtype=torch.long))''',
    "demo": """turns = [
    ('system', [1, 2]),
    ('user', [3, 4]),
    ('assistant', [10, 11, 12]),   # calls a tool
    ('tool', [20, 21, 22, 23]),    # tool output
    ('assistant', [13, 14]),       # final answer
]
for max_len in [16, 10]:
    ids, attn, loss = agent_rollout_tensors(turns, max_len, pad_id=0)
    print(f'max_len={max_len}')
    print('  input_ids     ', ids.tolist())
    print('  attention_mask', attn.tolist())
    print('  loss_mask     ', loss.tolist())""",

}
