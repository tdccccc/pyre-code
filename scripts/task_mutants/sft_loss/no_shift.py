# Mistake: scores each token with the logits at its own position, so labels are not shifted by one
def sft_loss(logits, input_ids, assistant_mask, seq_ids, reduction="token"):
    if reduction not in ("token", "sample"):
        raise ValueError(reduction)
    nll = torch.logsumexp(logits, dim=-1) - logits.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # BUG: no shift
    valid = assistant_mask.bool() & (seq_ids != -1)
    if not valid.any():
        return logits.sum() * 0.0
    if reduction == "token":
        return nll[valid].sum() / valid.sum()
    rows = torch.arange(nll.shape[0]).unsqueeze(1).expand_as(seq_ids)
    keys = rows[valid] * (int(seq_ids.max()) + 1) + seq_ids[valid]
    uniq, inv = torch.unique(keys, return_inverse=True)
    sums = nll.new_zeros(len(uniq)).index_add(0, inv, nll[valid])
    counts = nll.new_zeros(len(uniq)).index_add(0, inv, torch.ones_like(nll[valid]))
    return (sums / counts).mean()
