# Mistake: returns per-row dgamma / dbeta without summing over the leading dims, so their shape is wrong
import torch


def layernorm_backward(dy, x, gamma, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mu) * rstd

    dx_hat = dy * gamma
    dx = rstd * (dx_hat
                 - dx_hat.mean(dim=-1, keepdim=True)
                 - x_hat * (dx_hat * x_hat).mean(dim=-1, keepdim=True))

    dgamma = dy * x_hat  # BUG: forgot the sum over batch / sequence dims
    dbeta = dy
    return dx, dgamma, dbeta
