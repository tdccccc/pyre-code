# Mistake: uses the unbiased variance (divide by D-1) in rstd, so the normalization does not match the forward pass
import torch


def layernorm_backward(dy, x, gamma, eps=1e-5):
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).sum(dim=-1, keepdim=True) / (D - 1)  # BUG: unbiased
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mu) * rstd

    dx_hat = dy * gamma
    dx = rstd * (dx_hat
                 - dx_hat.mean(dim=-1, keepdim=True)
                 - x_hat * (dx_hat * x_hat).mean(dim=-1, keepdim=True))

    lead = tuple(range(dy.dim() - 1))
    dgamma = (dy * x_hat).sum(dim=lead)
    dbeta = dy.sum(dim=lead)
    return dx, dgamma, dbeta
