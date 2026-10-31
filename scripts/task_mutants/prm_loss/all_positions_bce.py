# Mistake: trains on every position, treating tokens that are not step tags as wrong steps (label 0)
def prm_loss(step_logits, input_ids, step_tag_id, step_labels, agg="min"):
    if agg not in ("min", "prod", "last"):
        raise ValueError(f"unknown agg: {agg}")
    is_tag = input_ids == step_tag_id
    # BUG: dense targets over all positions; non-tag positions become negatives
    targets = torch.zeros_like(step_logits)
    targets[is_tag] = step_labels[step_labels >= 0].to(step_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(step_logits, targets)
    with torch.no_grad():
        scores = []
        for i in range(step_logits.shape[0]):
            p = torch.sigmoid(step_logits[i][is_tag[i]])
            scores.append(p.min() if agg == "min" else p.prod() if agg == "prod" else p[-1])
        scores = torch.stack(scores)
    return loss, scores
