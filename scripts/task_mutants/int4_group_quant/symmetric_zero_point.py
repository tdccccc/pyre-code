# Mistake: symmetric quantization (absmax scale, fixed zero point 8) instead of asymmetric min/max
def int4_group_quantize(w, group_size):
    out_features, in_features = w.shape
    n_groups = in_features // group_size

    wg = w.reshape(out_features, n_groups, group_size)
    absmax = wg.abs().amax(dim=-1, keepdim=True)

    scale = absmax / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero = torch.full_like(scale, 8.0)

    q = torch.clamp(torch.round(wg / scale) + zero, 0, 15)
    q = q.reshape(out_features, in_features).to(torch.uint8)

    packed = q[:, 0::2] | (q[:, 1::2] << 4)

    scales = scale.squeeze(-1)
    zeros = zero.squeeze(-1).to(torch.uint8)
    return packed, scales, zeros
