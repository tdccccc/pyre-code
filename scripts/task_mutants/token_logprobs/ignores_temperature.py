# Mistake: ignores the sampling temperature, so training log-probs no longer match the rollout's
def token_logprobs(logits, input_ids, response_mask, temperature=1.0):
    logits = logits[:, :-1]  # BUG: never divided by temperature
    targets = input_ids[:, 1:]
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logps = picked - torch.logsumexp(logits, dim=-1)
    return logps * response_mask[:, 1:].to(logps.dtype)
