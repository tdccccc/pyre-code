# Mistake: derives the attention mask from input_ids != pad_id, which also hides real tokens that happen to equal pad_id
def agent_rollout_tensors(turns, max_len, pad_id):
    roles = ("system", "user", "assistant", "tool")
    ids, loss = [], []
    for role, token_ids in turns:
        if role not in roles:
            raise ValueError(f"unknown role: {role}")
        ids += token_ids
        loss += [1 if role == "assistant" else 0] * len(token_ids)
    if len(ids) > max_len:
        ids, loss = ids[:max_len], [0] * max_len
    n_pad = max_len - len(ids)
    ids_t = torch.tensor(ids + [pad_id] * n_pad, dtype=torch.long)
    # BUG: attention must come from the real length, not from the token values
    attn = (ids_t != pad_id).long()
    loss = loss + [0] * n_pad
    return ids_t, attn, torch.tensor(loss, dtype=torch.long)
