# Mistake: scores each token with its own position's logits instead of the previous position's, so the model "sees" the token it is predicting
def token_logprobs(logits, input_ids, response_mask, temperature=1.0):
    logits = logits[:, 1:] / temperature  # BUG: should be logits[:, :-1]
    targets = input_ids[:, 1:]
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logps = picked - torch.logsumexp(logits, dim=-1)
    return logps * response_mask[:, 1:].to(logps.dtype)
