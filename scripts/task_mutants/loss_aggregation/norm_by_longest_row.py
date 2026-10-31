# Mistake: with norm_len=None, divides by the longest valid row instead of the padded width T
def aggregate_loss(token_loss, mask, mode, norm_len=None):
    m = mask.to(token_loss.dtype)
    per_seq = (token_loss * m).sum(dim=1)
    if mode == "token-mean":
        return per_seq.sum() / m.sum()
    if mode == "seq-mean-token-mean":
        return (per_seq / m.sum(dim=1)).mean()
    if mode == "seq-mean-token-sum":
        return per_seq.mean()
    if mode == "seq-mean-token-sum-norm":
        # BUG: the default constant is T = token_loss.shape[1], not the longest row
        L = m.sum(dim=1).max() if norm_len is None else norm_len
        return per_seq.mean() / L
    raise ValueError(f"unknown mode: {mode}")
