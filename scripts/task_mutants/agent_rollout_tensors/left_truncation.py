# Mistake: keeps the last max_len tokens instead of the first, which drops the system prompt and the start of the conversation
def agent_rollout_tensors(turns, max_len, pad_id):
    roles = ("system", "user", "assistant", "tool")
    ids, loss = [], []
    for role, token_ids in turns:
        if role not in roles:
            raise ValueError(f"unknown role: {role}")
        ids += token_ids
        loss += [1 if role == "assistant" else 0] * len(token_ids)
    if len(ids) > max_len:
        # BUG: right truncation keeps ids[:max_len]
        ids, loss = ids[-max_len:], [0] * max_len
    n_pad = max_len - len(ids)
    attn = [1] * len(ids) + [0] * n_pad
    ids = ids + [pad_id] * n_pad
    loss = loss + [0] * n_pad
    return (torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long), torch.tensor(loss, dtype=torch.long))
