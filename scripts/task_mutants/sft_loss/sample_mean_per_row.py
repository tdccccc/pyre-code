# Mistake: the "sample" reduction averages per row, merging every sample packed into that row
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
    v = valid.to(nll.dtype)
    per_row_count = v.sum(dim=1)
    keep = per_row_count > 0
    per_row = (nll * v).sum(dim=1)[keep] / per_row_count[keep]  # BUG: one mean per row, not per (row, seq_id)
    return per_row.mean()
