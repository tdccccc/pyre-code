# Mistake: IPO puts beta on the margin, (beta*h - 1/2)^2, which is beta^2 times the real loss
def preference_loss(chosen_logps, rejected_logps, chosen_lens, rejected_lens, method,
                    ref_chosen_logps=None, ref_rejected_logps=None,
                    beta=0.1, label_smoothing=0.0, gamma=0.5):
    if method in ('ipo', 'cdpo'):
        if ref_chosen_logps is None or ref_rejected_logps is None:
            raise ValueError(f'{method} needs both reference log-probs')
        # Implicit reward margin, as in DPO; the reference model is a constant
        h = ((chosen_logps - ref_chosen_logps.detach())
             - (rejected_logps - ref_rejected_logps.detach()))
        if method == 'ipo':
            # Regress the margin to 1/(2*beta) instead of pushing it to infinity
            return ((beta * h - 0.5) ** 2).mean()  # BUG: should be (h - 1/(2*beta))**2
        z = beta * h
        eps = label_smoothing
        # The label is flipped with probability eps
        return (-(1 - eps) * F.logsigmoid(z) - eps * F.logsigmoid(-z)).mean()
    if method not in ('simpo', 'orpo'):
        raise ValueError(f'unknown method: {method!r}')

    # Reference-free methods use the average log-prob per token
    a_c = chosen_logps / chosen_lens.to(chosen_logps.dtype)
    a_r = rejected_logps / rejected_lens.to(rejected_logps.dtype)
    if method == 'simpo':
        return -F.logsigmoid(beta * (a_c - a_r) - gamma).mean()

    def log_odds(a):
        # log(p / (1 - p)) with p = exp(a); expm1 keeps 1 - p accurate when p is near 1
        return a - torch.log(-torch.expm1(a))

    nll = -a_c.mean()
    return nll + beta * (-F.logsigmoid(log_odds(a_c) - log_odds(a_r))).mean()
