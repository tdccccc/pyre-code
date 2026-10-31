# Mistake: computes log(softmax(x)), which underflows to -inf for large logits
def token_logprobs(logits, input_ids, response_mask, temperature=1.0):
    probs = torch.softmax(logits[:, :-1] / temperature, dim=-1)
    logps = torch.log(probs).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # BUG: unstable
    return logps * response_mask[:, 1:].to(logps.dtype)
