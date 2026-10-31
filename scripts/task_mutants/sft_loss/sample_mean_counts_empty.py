# Mistake: the "sample" reduction averages over every packed sample, counting samples with no assistant target as 0
def sft_loss(logits, input_ids, assistant_mask, seq_ids, reduction="token"):
    if reduction not in ("token", "sample"):
        raise ValueError(reduction)
    logits = logits[:, :-1]
    nll = torch.logsumexp(logits, dim=-1) - logits.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    cur, nxt = seq_ids[:, :-1], seq_ids[:, 1:]
    valid = assistant_mask[:, 1:].bool() & (nxt == cur) & (nxt != -1)
    if not valid.any():
        return logits.sum() * 0.0
    if reduction == "token":
        return nll[valid].sum() / valid.sum()
    real = nxt != -1  # BUG: groups every real (row, seq_id), even those with no counted target
    rows = torch.arange(nll.shape[0]).unsqueeze(1).expand_as(nxt)
    keys = rows[real] * (int(seq_ids.max()) + 1) + nxt[real]
    uniq, inv = torch.unique(keys, return_inverse=True)
    v = valid[real].to(nll.dtype)
    sums = nll.new_zeros(len(uniq)).index_add(0, inv, nll[real] * v)
    counts = nll.new_zeros(len(uniq)).index_add(0, inv, v)
    return (sums / counts.clamp(min=1)).mean()
