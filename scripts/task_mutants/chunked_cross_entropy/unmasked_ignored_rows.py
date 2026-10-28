# Mistake: ignored targets are clamped to class 0 but their loss is still summed, so ignored rows leak into the loss
def chunked_cross_entropy(hidden, weight, targets, chunk_size, ignore_index=-100):
    N = hidden.shape[0]
    total = hidden.new_zeros(())
    count = 0
    for start in range(0, N, chunk_size):
        h = hidden[start:start + chunk_size]
        t = targets[start:start + chunk_size]
        valid = t != ignore_index
        logits = h @ weight.T
        lse = torch.logsumexp(logits, dim=-1)
        safe_t = t.clamp(min=0)
        tgt_logit = logits.gather(1, safe_t[:, None])[:, 0]
        total = total + (lse - tgt_logit).sum()  # BUG: no valid mask on the sum
        count += int(valid.sum())
    return total / max(count, 1)
