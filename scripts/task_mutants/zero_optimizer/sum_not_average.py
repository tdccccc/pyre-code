# Mistake: reduce-scatter sums the rank gradients instead of averaging them
def zero2_adam_step(flat_param, rank_grads, opt_states, step, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    W = len(rank_grads)
    n = flat_param.numel() // W
    b1, b2 = betas

    new_parts, new_states = [], []
    for r in range(W):
        lo, hi = r * n, (r + 1) * n
        g = torch.stack([grad[lo:hi] for grad in rank_grads]).sum(dim=0)
        m = b1 * opt_states[r]["m"] + (1 - b1) * g
        v = b2 * opt_states[r]["v"] + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** step)
        v_hat = v / (1 - b2 ** step)
        new_parts.append(flat_param[lo:hi] - lr * m_hat / (v_hat.sqrt() + eps))
        new_states.append({"m": m, "v": v})

    new_flat = torch.cat(new_parts)
    return new_flat, new_states
