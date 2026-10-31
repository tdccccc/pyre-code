# Mistake: seq-mean-token-sum sums over sequences instead of averaging, so the loss scales with batch size
def aggregate_loss(token_loss, mask, mode, norm_len=None):
    m = mask.to(token_loss.dtype)
    per_seq = (token_loss * m).sum(dim=1)
    if mode == "token-mean":
        return per_seq.sum() / m.sum()
    if mode == "seq-mean-token-mean":
        return (per_seq / m.sum(dim=1)).mean()
    if mode == "seq-mean-token-sum":
        return per_seq.sum()  # BUG: should be per_seq.mean()
    if mode == "seq-mean-token-sum-norm":
        L = token_loss.shape[1] if norm_len is None else norm_len
        return per_seq.mean() / L
    raise ValueError(f"unknown mode: {mode}")
