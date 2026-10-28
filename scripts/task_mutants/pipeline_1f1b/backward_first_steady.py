# Mistake: in the steady state each stage runs B{j} before F{warmup+j}, swapping the pair
def pipeline_1f1b_schedule(num_stages, num_microbatches):
    p, m = num_stages, num_microbatches
    schedule = []
    for s in range(p):
        warmup = min(p - s - 1, m)
        ops = [f"F{i}" for i in range(warmup)]
        for j in range(m - warmup):
            ops.append(f"B{j}")
            ops.append(f"F{warmup + j}")
        ops += [f"B{j}" for j in range(m - warmup, m)]
        schedule.append(ops)
    bubble_fraction = (p - 1) / (m + p - 1)
    return schedule, bubble_fraction
