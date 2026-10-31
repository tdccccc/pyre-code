# Mistake: takes the mask from the input position instead of the target position, so the last prompt token's prediction is kept and the last response token's is dropped
def token_logprobs(logits, input_ids, response_mask, temperature=1.0):
    logits = logits[:, :-1] / temperature
    targets = input_ids[:, 1:]
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    logps = picked - torch.logsumexp(logits, dim=-1)
    return logps * response_mask[:, :-1].to(logps.dtype)  # BUG: should be response_mask[:, 1:]
