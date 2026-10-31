# Mistake: divides the log-probabilities by the temperature instead of the logits, which is not a distribution any more
def token_logprobs(logits, input_ids, response_mask, temperature=1.0):
    lp = torch.log_softmax(logits[:, :-1], dim=-1) / temperature  # BUG: temperature belongs inside the softmax
    logps = lp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logps * response_mask[:, 1:].to(logps.dtype)
