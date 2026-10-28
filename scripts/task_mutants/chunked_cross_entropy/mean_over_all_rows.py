# Mistake: divides by the total number of rows, so ignored rows still dilute the mean
def chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100):
    N = hidden.shape[0]
    total = hidden.new_zeros(())
    for start in range(0, N, chunk_size):
        h = hidden[start:start + chunk_size]
        t = targets[start:start + chunk_size]
        valid = t != ignore_index
        logits = h @ weight.T
        lse = torch.logsumexp(logits, dim=-1)
        safe_t = t.masked_fill(~valid, 0)
        tgt_logit = logits.gather(1, safe_t[:, None])[:, 0]
        total = total + ((lse - tgt_logit) * valid).sum()
    return total / max(N, 1)  # BUG: should divide by the count of non-ignored rows
