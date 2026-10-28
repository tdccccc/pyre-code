# Mistake: admission skips a request that does not fit and admits a later smaller one (greedy, not strict FCFS)
def schedule_continuous_batching(requests, max_batch_tokens, max_num_seqs):
    for _, prompt_len, _ in requests:
        if prompt_len > max_batch_tokens:
            raise ValueError(f"prompt_len {prompt_len} can never fit in max_batch_tokens={max_batch_tokens}")
    # waiting requests in FCFS order: by arrival step, then by id
    waiting = sorted(range(len(requests)), key=lambda i: (requests[i][0], i))
    left = [out_len for _, _, out_len in requests]  # output tokens still to produce
    running = []  # admitted and not finished
    steps = []
    step = 0
    while any(left):
        # (1) every running request decodes one token
        ran = list(running)
        tokens = len(running)
        # (2) admit arrived requests in FCFS order; stop at the first one that does not fit
        admitted = []
        for rid in list(waiting):
            if requests[rid][0] > step:
                break
            prompt_len = requests[rid][1]
            if len(running) + len(admitted) + 1 > max_num_seqs:
                break
            if tokens + prompt_len > max_batch_tokens:
                continue  # skip ahead to a smaller request
            waiting.remove(rid)
            admitted.append(rid)
            tokens += prompt_len
        ran += admitted
        # every request that ran in this step produces one output token
        for rid in ran:
            left[rid] -= 1
        steps.append(sorted(ran))
        # requests that produced their last token leave after this step
        running = [rid for rid in ran if left[rid] > 0]
        step += 1
    return steps
