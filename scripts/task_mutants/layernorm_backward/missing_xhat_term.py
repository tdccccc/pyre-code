# Mistake: drops the x_hat * mean(dx_hat * x_hat) term, so the gradient ignores how the mean and variance depend on x
import torch


def layernorm_backward(dy, x, gamma, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mu) * rstd

    dx_hat = dy * gamma
    dx = rstd * (dx_hat - dx_hat.mean(dim=-1, keepdim=True))  # BUG: missing - x_hat * mean(dx_hat * x_hat)

    lead = tuple(range(dy.dim() - 1))
    dgamma = (dy * x_hat).sum(dim=lead)
    dbeta = dy.sum(dim=lead)
    return dx, dgamma, dbeta
