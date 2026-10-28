# Mistake: ignores the eps argument and always uses 1e-5 inside the square root
import torch


def layernorm_backward(dy, x, gamma, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + 1e-5)  # BUG: should use eps
    x_hat = (x - mu) * rstd

    dx_hat = dy * gamma
    dx = rstd * (dx_hat
                 - dx_hat.mean(dim=-1, keepdim=True)
                 - x_hat * (dx_hat * x_hat).mean(dim=-1, keepdim=True))

    lead = tuple(range(dy.dim() - 1))
    dgamma = (dy * x_hat).sum(dim=lead)
    dbeta = dy.sum(dim=lead)
    return dx, dgamma, dbeta
