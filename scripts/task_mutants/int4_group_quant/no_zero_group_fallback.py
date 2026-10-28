# Mistake: no scale == 0 fallback, so an all-zero group divides by zero
def int4_group_quantize(w, group_size):
    out_features, in_features = w.shape
    n_groups = in_features // group_size

    wg = w.reshape(out_features, n_groups, group_size)
    mn = wg.amin(dim=-1, keepdim=True)
    mx = wg.amax(dim=-1, keepdim=True)

    scale = (mx - mn) / 15.0
    zero = torch.clamp(torch.round(-mn / scale), 0, 15)

    q = torch.clamp(torch.round(wg / scale) + zero, 0, 15)
    q = q.reshape(out_features, in_features).to(torch.uint8)

    packed = q[:, 0::2] | (q[:, 1::2] << 4)

    scales = scale.squeeze(-1)
    zeros = zero.squeeze(-1).to(torch.uint8)
    return packed, scales, zeros
