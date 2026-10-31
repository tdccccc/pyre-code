# Mistake: defaults to the product of step probabilities, which keeps shrinking with the number of steps, instead of the minimum
def prm_loss(step_logits, input_ids, step_tag_id, step_labels, agg="prod"):
    # BUG: the default aggregation should be "min"
    if agg not in ("min", "prod", "last"):
        raise ValueError(f"unknown agg: {agg}")
    is_tag = input_ids == step_tag_id
    logits = step_logits[is_tag]
    labels = step_labels[step_labels >= 0].to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    with torch.no_grad():
        scores = []
        for i in range(step_logits.shape[0]):
            p = torch.sigmoid(step_logits[i][is_tag[i]])
            scores.append(p.min() if agg == "min" else p.prod() if agg == "prod" else p[-1])
        scores = torch.stack(scores)
    return loss, scores
