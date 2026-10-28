"""Continuous Batching Scheduler task."""

TASK = {
    "title": "Continuous Batching Scheduler",
    "title_zh": "连续批处理调度器",
    "difficulty": "Hard",
    "description_en": "Implement iteration-level (continuous) batching as in Orca and vLLM. Requests join and leave the running batch at every step, instead of waiting for the whole batch to finish. This keeps the GPU busy and lowers the latency of short requests.\n\nTime advances in discrete steps 0, 1, 2, ... Each request runs one prefill step (its whole prompt, producing its first output token), then one decode step per further output token.\n\n**Signature:** `schedule_continuous_batching(requests, max_batch_tokens, max_num_seqs) -> list[list[int]]`\n\n**Parameters:**\n- `requests` — list of `(arrival_step, prompt_len, output_len)`; the request id is its index in the list\n- `max_batch_tokens` — maximum number of tokens processed in one step\n- `max_num_seqs` — maximum number of requests in the running batch\n\n**Returns:** for each simulated step, in order, the sorted list of request ids that ran in that step (`[]` for an idle step). The simulation stops after the step in which the last request finishes.\n\n**Step rules.** Step t does the following:\n1. Every request running at the start of step t (admitted in an earlier step and not finished) decodes one token. This costs 1 token per request.\n2. Waiting requests with `arrival_step <= t` are admitted in FCFS order (by `arrival_step`, then id). A request is admitted only if the running requests after admission would be at most `max_num_seqs`, and the step's token total (1 per decode plus `prompt_len` of each admitted request) would stay at most `max_batch_tokens`. Admission is strictly FCFS: stop at the first arrived request that does not fit, and do not skip ahead to smaller ones.\n3. An admitted request runs its prefill in step t. This costs `prompt_len` tokens and produces its first output token. A running request produces one output token per step.\n4. A request is finished at the end of the step in which it has produced `output_len` tokens, and it does not run afterwards. A request that finishes in step t still occupies its batch slot and its tokens during step t, so it counts for `max_num_seqs` in step t's admission. Its slot is free from step t+1 on.\n\n**Example:** `schedule_continuous_batching([(0, 4, 2), (1, 2, 1)], 8, 2)` returns `[[0], [0, 1]]`.\n\n**Constraints:**\n- `prompt_len >= 1`, `output_len >= 1`, `max_num_seqs >= 1`\n- If any `prompt_len > max_batch_tokens`, raise `ValueError`, since that request could never be scheduled\n- With no requests, return `[]`",
    "description_zh": "实现 Orca 和 vLLM 中的迭代级（连续）批处理。请求在每一步都可以加入或离开运行批次，而不是等整批全部结束。这样能保持 GPU 忙碌，并降低短请求的延迟。\n\n时间以离散步骤 0, 1, 2, ... 推进。每个请求先执行一次 prefill 步（处理整个提示并产生第一个输出 token），之后每多生成一个输出 token 执行一次 decode 步。\n\n**签名:** `schedule_continuous_batching(requests, max_batch_tokens, max_num_seqs) -> list[list[int]]`\n\n**参数:**\n- `requests` — `(arrival_step, prompt_len, output_len)` 的列表；请求编号为其在列表中的下标\n- `max_batch_tokens` — 一步中最多处理的 token 数\n- `max_num_seqs` — 运行批次中最多的请求数\n\n**返回:** 按顺序给出每个模拟步骤中实际运行的请求编号（已排序）的列表；空闲步骤为 `[]`。模拟在最后一个请求结束的那一步之后停止。\n\n**步骤规则。** 第 t 步依次执行：\n1. 第 t 步开始时正在运行的每个请求（之前某步已准入且未结束）decode 一个 token，每个请求花费 1 个 token。\n2. 按 FCFS 顺序（先按 `arrival_step`，再按编号）准入 `arrival_step <= t` 的等待请求。只有准入后运行中的请求数不超过 `max_num_seqs`，且本步 token 总数（每个 decode 计 1，加上每个新准入请求的 `prompt_len`）仍不超过 `max_batch_tokens` 时，才准入该请求。准入严格遵循 FCFS：遇到第一个已到达但放不下的请求就停止，不跳过它去准入更小的请求。\n3. 被准入的请求在第 t 步执行 prefill，花费 `prompt_len` 个 token，并产生第一个输出 token。运行中的请求每步产生一个输出 token。\n4. 请求在它产生第 `output_len` 个输出 token 的那一步结束时完成，之后不再运行。在第 t 步结束的请求，在第 t 步期间仍占用批次槽位和 token，因此在第 t 步的准入判断中仍计入 `max_num_seqs`；从第 t+1 步起释放该槽位。\n\n**示例:** `schedule_continuous_batching([(0, 4, 2), (1, 2, 1)], 8, 2)` 返回 `[[0], [0, 1]]`。\n\n**约束:**\n- `prompt_len >= 1`，`output_len >= 1`，`max_num_seqs >= 1`\n- 若任一 `prompt_len > max_batch_tokens`，抛出 `ValueError`，因为该请求永远无法被调度\n- 没有请求时返回 `[]`",
    "function_name": "schedule_continuous_batching",
    "hint": "waiting = request ids sorted by (arrival_step, id);  left[i] = output tokens still to produce\nrunning = []\nfor t = 0, 1, 2, ... while any(left) > 0:\n  ran = running;  tokens = len(running)           # 1) decode\n  admitted = []                                     # 2) FCFS admission\n  while waiting[0] has arrived:\n    if len(running) + len(admitted) + 1 > max_num_seqs: break\n    if tokens + prompt_len > max_batch_tokens: break   # strict FCFS: no skipping\n    pop it; admitted.append; tokens += prompt_len\n  for each id in ran + admitted: left[id] -= 1     # one token each\n  record sorted(ran + admitted)\n  running = [id in ran + admitted if left[id] > 0]",
    "hint_zh": "waiting = 按 (arrival_step, 编号) 排序的请求编号;  left[i] = 尚需生成的输出 token 数\nrunning = []\nfor t = 0, 1, 2, ... 只要 any(left) > 0:\n  ran = running;  tokens = len(running)            # 1) decode\n  admitted = []                                     # 2) FCFS 准入\n  while waiting[0] 已到达:\n    if len(running) + len(admitted) + 1 > max_num_seqs: break\n    if tokens + prompt_len > max_batch_tokens: break   # 严格 FCFS：不跳过\n    弹出该请求; admitted.append; tokens += prompt_len\n  对 ran + admitted 中每个编号: left[id] -= 1     # 每个生成一个 token\n  记录 sorted(ran + admitted)\n  running = [ran + admitted 中 left[id] > 0 的编号]",
    "tests": [
        {
            "name": "Single request: prefill then decodes",
            "code": """
got = {fn}([(0, 4, 3)], 100, 4)
expected = [[0], [0], [0]]
assert got == expected, f'prefill in step 0, then one decode per step: expected {expected}, got {got}'
""",
        },
        {
            "name": "Late arrival causes idle steps",
            "code": """
got = {fn}([(3, 2, 1), (0, 2, 2)], 100, 4)
expected = [[1], [1], [], [0]]
assert got == expected, f'request 1 runs at steps 0-1, steps 2 is idle, request 0 arrives at 3: expected {expected}, got {got}'
""",
        },
        {
            "name": "Token budget delays admission",
            "code": """
got = {fn}([(0, 6, 2), (0, 5, 2)], 10, 8)
expected = [[0], [0, 1], [1]]
assert got == expected, f'6 + 5 > 10 delays request 1 to step 1, where the decode of request 0 costs 1: expected {expected}, got {got}'
""",
        },
        {
            "name": "Decode tokens count against the budget",
            "code": """
got = {fn}([(0, 3, 4), (1, 4, 1)], 4, 8)
expected = [[0], [0], [0], [0], [1]]
assert got == expected, f'request 1 needs 4 tokens but the running decode leaves only 3 of budget 4: expected {expected}, got {got}'
""",
        },
        {
            "name": "max_num_seqs caps the batch, finished requests free slots",
            "code": """
got = {fn}([(0, 1, 3), (0, 1, 3), (0, 1, 3)], 100, 2)
expected = [[0, 1], [0, 1], [0, 1], [2], [2], [2]]
assert got == expected, f'request 2 waits until step 3 because both slots are held through step 2: expected {expected}, got {got}'
""",
        },
        {
            "name": "Strict FCFS: a small request does not jump a blocked large one",
            "code": """
got = {fn}([(0, 2, 1), (0, 9, 1), (0, 1, 1)], 10, 10)
expected = [[0], [1, 2]]
assert got == expected, f'request 2 would fit in step 0, but it waits behind request 1: expected {expected}, got {got}'
""",
        },
        {
            "name": "Requests leaving free capacity for later arrivals",
            "code": """
got = {fn}([(0, 1, 5), (0, 1, 1), (0, 1, 1)], 100, 2)
expected = [[0, 1], [0, 2], [0], [0], [0]]
assert got == expected, f'request 1 finishes in step 0, so request 2 is admitted in step 1: expected {expected}, got {got}'
""",
        },
        {
            "name": "Empty workload and oversized prompt",
            "code": """
assert {fn}([], 8, 2) == [], 'no requests means no steps'
try:
    {fn}([(0, 9, 1)], 8, 2)
except ValueError:
    pass
else:
    raise AssertionError('a prompt longer than max_batch_tokens must raise ValueError')
""",
        },
        {
            "name": "Randomized comparison with an independent simulation",
            "code": """
import random

def reference(requests, budget, max_seqs):
    n = len(requests)
    produced = [0] * n
    admitted = [False] * n
    finished = [False] * n
    out = []
    t = 0
    while not all(finished):
        step_ids = [i for i in range(n) if admitted[i] and not finished[i]]
        slots_used = len(step_ids)
        tokens = len(step_ids)
        order = sorted((i for i in range(n) if not admitted[i]), key=lambda i: (requests[i][0], i))
        for i in order:
            arrival, prompt_len, _ = requests[i]
            if arrival > t or slots_used + 1 > max_seqs or tokens + prompt_len > budget:
                break
            admitted[i] = True
            slots_used += 1
            tokens += prompt_len
            step_ids.append(i)
        for i in step_ids:
            produced[i] += 1
            if produced[i] == requests[i][2]:
                finished[i] = True
        out.append(sorted(step_ids))
        t += 1
    return out

rng = random.Random(7)
for trial in range(300):
    n = rng.randint(1, 12)
    requests = [(rng.randint(0, 15), rng.randint(1, 8), rng.randint(1, 6)) for _ in range(n)]
    budget = rng.randint(8, 20)
    max_seqs = rng.randint(1, 5)
    expected = reference(requests, budget, max_seqs)
    got = {fn}(requests, budget, max_seqs)
    assert got == expected, f'trial {trial}: requests={requests} budget={budget} max_seqs={max_seqs}\\nexpected {expected}\\ngot      {got}'
""",
        },
    ],
    "solution": '''def schedule_continuous_batching(requests, max_batch_tokens, max_num_seqs):
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
        while waiting and requests[waiting[0]][0] <= step:
            rid = waiting[0]
            prompt_len = requests[rid][1]
            if len(running) + len(admitted) + 1 > max_num_seqs:
                break
            if tokens + prompt_len > max_batch_tokens:
                break
            waiting.pop(0)
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
    return steps''',
    "demo": """requests = [(0, 6, 3), (1, 4, 2), (2, 2, 1), (9, 3, 2)]  # (arrival, prompt_len, output_len)
steps = schedule_continuous_batching(requests, max_batch_tokens=10, max_num_seqs=3)
for t, ids in enumerate(steps):
    print(f'step {t:2d}: running {ids}')""",
}
