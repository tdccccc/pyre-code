# Mistake: warmup = min(num_stages - s, m), one forward more than needed, so in-flight activations exceed min(p - s, m)
def pipeline_1f1b_schedule(num_stages, num_microbatches):
    p, m = num_stages, num_microbatches
    schedule = []
    for s in range(p):
        warmup = min(p - s, m)
        ops = [f"F{i}" for i in range(warmup)]
        for j in range(m - warmup):
            ops.append(f"F{warmup + j}")
            ops.append(f"B{j}")
        ops += [f"B{j}" for j in range(m - warmup, m)]
        schedule.append(ops)
    bubble_fraction = (p - 1) / (m + p - 1)
    return schedule, bubble_fraction
