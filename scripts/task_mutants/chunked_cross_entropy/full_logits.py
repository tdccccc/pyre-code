# Mistake: materializes the full (N, V) logits up front, so chunking saves no memory
def chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100):
    N = hidden.shape[0]
    logits = hidden @ weight.T  # BUG: whole (N, V) block
    total = hidden.new_zeros(())
    count = 0
    for start in range(0, N, chunk_size):
        lg = logits[start:start + chunk_size]
        t = targets[start:start + chunk_size]
        valid = t != ignore_index
        lse = torch.logsumexp(lg, dim=-1)
        safe_t = t.masked_fill(~valid, 0)
        tgt_logit = lg.gather(1, safe_t[:, None])[:, 0]
        total = total + ((lse - tgt_logit) * valid).sum()
        count += int(valid.sum())
    return total / max(count, 1)
