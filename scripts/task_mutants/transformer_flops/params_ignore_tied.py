# Mistake: always counts a separate LM head matrix, ignoring tied_embeddings in the parameter count
def transformer_flops(n_layers, d_model, d_ff, vocab_size, seq_len, batch_size, tied_embeddings=True):
    d, V, L = d_model, vocab_size, n_layers
    tokens = batch_size * seq_len
    embed = V * d
    lm_head = V * d
    per_layer = 4 * d * d + 2 * d * d_ff + 4 * d
    params = embed + lm_head + L * per_layer + 2 * d
    weight_flops = 2 * tokens * (L * (4 * d * d + 2 * d * d_ff) + V * d)
    attn_flops = L * 4 * batch_size * seq_len * seq_len * d
    forward = weight_flops + attn_flops
    return {"params": params, "forward_flops": forward, "train_flops": 3 * forward}
