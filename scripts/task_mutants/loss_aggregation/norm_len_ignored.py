# Mistake: seq-mean-token-sum-norm always divides by the padded length T and ignores norm_len
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
        return per_seq.mean() / token_loss.shape[1]  # BUG: norm_len is ignored
    raise ValueError(f"unknown mode: {mode}")
