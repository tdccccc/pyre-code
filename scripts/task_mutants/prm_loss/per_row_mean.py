# Mistake: averages the BCE within each solution first and then over solutions, so a step in a short solution weighs more than a step in a long one
def prm_loss(step_logits, input_ids, step_tag_id, step_labels, agg="min"):
    if agg not in ("min", "prod", "last"):
        raise ValueError(f"unknown agg: {agg}")
    is_tag = input_ids == step_tag_id
    row_losses = []
    for i in range(step_logits.shape[0]):
        lg = step_logits[i][is_tag[i]]
        lb = step_labels[i][step_labels[i] >= 0].to(lg.dtype)
        row_losses.append(F.binary_cross_entropy_with_logits(lg, lb))
    # BUG: mean of per-row means instead of one mean over all steps
    loss = torch.stack(row_losses).mean()
    with torch.no_grad():
        scores = []
        for i in range(step_logits.shape[0]):
            p = torch.sigmoid(step_logits[i][is_tag[i]])
            scores.append(p.min() if agg == "min" else p.prod() if agg == "prod" else p[-1])
        scores = torch.stack(scores)
    return loss, scores
