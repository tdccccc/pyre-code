# Mistake: averages the divergence within each sequence first and then over sequences, so tokens of short sequences weigh more than the batch token-mean
def distill_loss(student_logits, teacher_logits, mask, mode="forward", temperature=1.0, beta=0.5):
    if mode not in ("forward", "reverse", "jsd"):
        raise ValueError(f"unknown mode: {mode}")
    if mode == "jsd" and not 0.0 < beta < 1.0:
        raise ValueError("beta must be in (0, 1)")
    log_ps = torch.log_softmax(student_logits / temperature, dim=-1)
    log_pt = torch.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    if mode == "forward":
        div = (log_pt.exp() * (log_pt - log_ps)).sum(-1)
    elif mode == "reverse":
        div = (log_ps.exp() * (log_ps - log_pt)).sum(-1)
    else:
        log_m = torch.logsumexp(torch.stack([log_pt + math.log(beta), log_ps + math.log(1 - beta)]), dim=0)
        div = (beta * (log_pt.exp() * (log_pt - log_m)).sum(-1)
               + (1 - beta) * (log_ps.exp() * (log_ps - log_m)).sum(-1))
    m = mask.to(div.dtype)
    # BUG: per-row mean, then mean over rows
    return temperature ** 2 * ((div * m).sum(-1) / m.sum(-1)).mean()
