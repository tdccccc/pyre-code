# Mistake: computes BCE as log(sigmoid(x)) and log(1 - sigmoid(x)), which gives inf once sigmoid saturates to exactly 0 or 1
def prm_loss(step_logits, input_ids, step_tag_id, step_labels, agg="min"):
    if agg not in ("min", "prod", "last"):
        raise ValueError(f"unknown agg: {agg}")
    is_tag = input_ids == step_tag_id
    logits = step_logits[is_tag]
    labels = step_labels[step_labels >= 0].to(logits.dtype)
    p = torch.sigmoid(logits)
    # BUG: log(1 - 1.0) = -inf for a large positive logit
    loss = -(labels * torch.log(p) + (1 - labels) * torch.log(1 - p)).mean()
    with torch.no_grad():
        scores = []
        for i in range(step_logits.shape[0]):
            q = torch.sigmoid(step_logits[i][is_tag[i]])
            scores.append(q.min() if agg == "min" else q.prod() if agg == "prod" else q[-1])
        scores = torch.stack(scores)
    return loss, scores
