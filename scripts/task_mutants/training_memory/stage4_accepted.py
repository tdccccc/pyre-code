# Mistake: no validation of zero_stage, so stage 4 is silently treated like stage 3
def training_memory(num_params, n_layers, d_model, n_heads, seq_len, micro_batch, zero_stage=0, world_size=1, recompute=False):
    param_div = world_size if zero_stage >= 3 else 1
    grad_div = world_size if zero_stage >= 2 else 1
    opt_div = world_size if zero_stage >= 1 else 1
    params = 2 * num_params / param_div
    grads = 2 * num_params / grad_div
    optimizer = 12 * num_params / opt_div
    s, b, h, a = seq_len, micro_batch, d_model, n_heads
    if recompute:
        per_layer = 2 * s * b * h
    else:
        per_layer = s * b * h * (34 + 5 * a * s / h)
    activations = float(n_layers * per_layer)
    total = params + grads + optimizer + activations
    return {
        "params": float(params),
        "grads": float(grads),
        "optimizer": float(optimizer),
        "activations": activations,
        "total": float(total),
    }
