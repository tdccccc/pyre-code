# Mistake: reads the assistant mask at the input position instead of the target position, so the last prompt token's prediction is trained and the final assistant token is not
def sft_loss(logits, input_ids, assistant_mask, seq_ids, reduction="token"):
    if reduction not in ("token", "sample"):
        raise ValueError(reduction)
    logits = logits[:, :-1]
    nll = torch.logsumexp(logits, dim=-1) - logits.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    cur, nxt = seq_ids[:, :-1], seq_ids[:, 1:]
    valid = assistant_mask[:, :-1].bool() & (nxt == cur) & (nxt != -1)  # BUG: should be assistant_mask[:, 1:]
    if not valid.any():
        return logits.sum() * 0.0
    if reduction == "token":
        return nll[valid].sum() / valid.sum()
    rows = torch.arange(nll.shape[0]).unsqueeze(1).expand_as(nxt)
    keys = rows[valid] * (int(seq_ids.max()) + 1) + nxt[valid]
    uniq, inv = torch.unique(keys, return_inverse=True)
    sums = nll.new_zeros(len(uniq)).index_add(0, inv, nll[valid])
    counts = nll.new_zeros(len(uniq)).index_add(0, inv, torch.ones_like(nll[valid]))
    return (sums / counts).mean()
