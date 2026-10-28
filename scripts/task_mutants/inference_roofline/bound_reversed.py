# Mistake: bound rule reversed, calling low-intensity steps compute bound and high-intensity steps memory bound
def inference_roofline(num_params, n_layers, d_model, n_heads, n_kv_heads, seq_len, batch_size, peak_flops, mem_bandwidth, bytes_per_param=2):
    head_dim = d_model // n_heads
    kv_cache_bytes = 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_param
    flops = 2 * num_params * batch_size
    total_bytes = num_params * bytes_per_param + kv_cache_bytes
    intensity = flops / total_bytes
    ridge = peak_flops / mem_bandwidth
    bound = "compute" if intensity < ridge else "memory"
    step_time = max(flops / peak_flops, total_bytes / mem_bandwidth)
    tokens_per_sec = batch_size / step_time
    mfu = flops / step_time / peak_flops
    return {
        "kv_cache_bytes": kv_cache_bytes,
        "flops": flops,
        "bytes": total_bytes,
        "arithmetic_intensity": intensity,
        "bound": bound,
        "step_time": step_time,
        "tokens_per_sec": tokens_per_sec,
        "mfu": mfu,
    }
