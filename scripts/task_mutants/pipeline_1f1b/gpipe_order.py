# Mistake: GPipe order (all forwards, then all backwards), so every stage keeps all m micro-batches in flight
def pipeline_1f1b_schedule(num_stages, num_microbatches):
    p, m = num_stages, num_microbatches
    schedule = []
    for s in range(p):
        ops = [f"F{i}" for i in range(m)] + [f"B{i}" for i in range(m)]
        schedule.append(ops)
    bubble_fraction = (p - 1) / (m + p - 1)
    return schedule, bubble_fraction
