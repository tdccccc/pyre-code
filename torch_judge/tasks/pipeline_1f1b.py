"""Pipeline Parallel 1F1B Schedule task."""

TASK = {
    "title": "Pipeline Parallel 1F1B Schedule",
    "title_zh": "流水线并行 1F1B 调度",
    "difficulty": "Medium",
    "description_en": "Implement the 1F1B (one-forward-one-backward) pipeline-parallel schedule used by PipeDream-Flush and Megatron-LM.\n\nA model is split into $p$ pipeline stages, and each mini-batch is split into $m$ micro-batches. GPipe runs all forwards and then all backwards, so every stage keeps the activations of all $m$ micro-batches alive at once. 1F1B starts each backward as early as possible: after a warmup of forwards, every stage alternates one forward with one backward. This bounds the in-flight activations on stage $s$ to $\\min(p - s, m)$ micro-batches instead of $m$, which is what makes large $m$ affordable in memory. With equal forward and backward times, the ideal pipeline bubble fraction is $\\frac{p - 1}{m + p - 1}$.\n\n**Signature:** `pipeline_1f1b_schedule(num_stages, num_microbatches) -> tuple[list[list[str]], float]`\n\n**Parameters:**\n- `num_stages` — $p \\ge 1$, the number of pipeline stages (stage 0 is the first)\n- `num_microbatches` — $m \\ge 1$, the number of micro-batches (indices are 0-based)\n\n**Returns:** `(schedule, bubble_fraction)`\n- `schedule[s]` — the ordered list of operations run by stage $s$. Each operation is a string: `\"F{i}\"` for the forward of micro-batch $i$, or `\"B{i}\"` for its backward. Every stage runs each `F{i}` and each `B{i}` exactly once.\n- `bubble_fraction` — $\\frac{p - 1}{m + p - 1}$, as a float.\n\n**Algorithm:** (for stage $s$, with $\\texttt{warmup} = \\min(p - s - 1, m)$):\n```\n1. warmup:    F0, F1, ..., F{warmup-1}\n2. steady:    for j = 0 .. m - warmup - 1:  append F{warmup+j}, then B{j}\n3. cooldown:  B{m-warmup}, B{m-warmup+1}, ..., B{m-1}\n```\n\n**Constraints:**\n- On every stage, `F{i}` comes before `B{i}`.\n- Dependencies: `F{i}` on stage $s$ needs `F{i}` finished on stage $s-1$; `B{i}` on stage $s$ needs `B{i}` finished on stage $s+1$. Executing the schedule with unit-time operations must never deadlock.\n- The peak number of in-flight micro-batches on stage $s$ (forwards run minus backwards run, maximized over time) is $\\min(p - s, m)$. GPipe would keep $m$ micro-batches in flight on every stage.\n- When $m < p$, the warmup is capped at $m$, so the schedule still contains exactly $m$ forwards and $m$ backwards per stage.",
    "description_zh": "实现 1F1B（一前向一反向）流水线并行调度，即 PipeDream-Flush 和 Megatron-LM 中使用的调度方式。\n\n模型被切分为 $p$ 个流水线阶段（stage），每个 mini-batch 被切分为 $m$ 个微批次（micro-batch）。GPipe 先执行全部前向，再执行全部反向，因此每个 stage 必须同时保留所有 $m$ 个微批次的激活值。1F1B 尽早开始每个反向：先做一段前向热身（warmup），之后每个 stage 交替执行一次前向和一次反向。这样，stage $s$ 上同时在途的微批次数量被限制为 $\\min(p - s, m)$，而不是 $m$，从而使较大的 $m$ 在显存上可以承受。在前向与反向耗时相等的情况下，理想的流水线气泡比例为 $\\frac{p - 1}{m + p - 1}$。\n\n**签名:** `pipeline_1f1b_schedule(num_stages, num_microbatches) -> tuple[list[list[str]], float]`\n\n**参数:**\n- `num_stages` — $p \\ge 1$，流水线 stage 数量（stage 0 为第一个）\n- `num_microbatches` — $m \\ge 1$，微批次数量（下标从 0 开始）\n\n**返回:** `(schedule, bubble_fraction)`\n- `schedule[s]` — stage $s$ 按顺序执行的操作列表。每个操作是一个字符串：微批次 $i$ 的前向为 `\"F{i}\"`，其反向为 `\"B{i}\"`。每个 stage 中每个 `F{i}` 和每个 `B{i}` 都恰好出现一次。\n- `bubble_fraction` — $\\frac{p - 1}{m + p - 1}$，浮点数。\n\n**算法:**（对 stage $s$，令 $\\texttt{warmup} = \\min(p - s - 1, m)$）\n```\n1. 热身:    F0, F1, ..., F{warmup-1}\n2. 稳态:    对 j = 0 .. m - warmup - 1：依次追加 F{warmup+j}，然后 B{j}\n3. 收尾:    B{m-warmup}, B{m-warmup+1}, ..., B{m-1}\n```\n\n**约束:**\n- 在每个 stage 上，`F{i}` 必须排在 `B{i}` 之前。\n- 依赖关系：stage $s$ 上的 `F{i}` 需要 stage $s-1$ 上的 `F{i}` 已完成；stage $s$ 上的 `B{i}` 需要 stage $s+1$ 上的 `B{i}` 已完成。以单位时间执行该调度时不得死锁。\n- stage $s$ 上同时在途的微批次峰值数量（已执行的前向数减去已执行的反向数，在整个执行过程中取最大值）为 $\\min(p - s, m)$。GPipe 在每个 stage 上都会保留 $m$ 个在途微批次。\n- 当 $m < p$ 时，热身长度上限为 $m$，因此每个 stage 中前向有 $m$ 个、反向也有 $m$ 个。",
    "function_name": "pipeline_1f1b_schedule",
    "hint": "for s in range(p):\n    warmup = min(p - s - 1, m)\n    ops = [F0..F{warmup-1}]\n    for j in range(m - warmup): ops += [F{warmup+j}, B{j}]\n    ops += [B{j} for j in range(m - warmup, m)]\nbubble = (p - 1) / (m + p - 1)",
    "hint_zh": "for s in range(p):\n    warmup = min(p - s - 1, m)\n    ops = [F0..F{warmup-1}]\n    for j in range(m - warmup): ops += [F{warmup+j}, B{j}]\n    ops += [B{j} for j in range(m - warmup, m)]\nbubble = (p - 1) / (m + p - 1)",
    "tests": [
        {
            "name": "Exact schedule for 4 stages, 8 micro-batches",
            "code": """
import torch
sched, _ = {fn}(4, 8)
assert len(sched) == 4, f'Expected 4 stages, got {len(sched)}'
expected_s0 = ['F0', 'F1', 'F2', 'F3', 'B0', 'F4', 'B1', 'F5', 'B2', 'F6', 'B3', 'F7', 'B4', 'B5', 'B6', 'B7']
expected_s3 = ['F0', 'B0', 'F1', 'B1', 'F2', 'B2', 'F3', 'B3', 'F4', 'B4', 'F5', 'B5', 'F6', 'B6', 'F7', 'B7']
assert sched[0] == expected_s0, f'Stage 0: got {sched[0]}'
assert sched[3] == expected_s3, f'Stage 3: got {sched[3]}'
""",
        },
        {
            "name": "Exact schedule for 2 stages, 2 micro-batches",
            "code": """
import torch
sched, _ = {fn}(2, 2)
assert sched == [['F0', 'F1', 'B0', 'B1'], ['F0', 'B0', 'F1', 'B1']], f'Got {sched}'
""",
        },
        {
            "name": "Every stage runs each F and B once, F before B",
            "code": """
import torch
for p, m in [(4, 8), (3, 5), (5, 3), (1, 4), (6, 6), (2, 7)]:
    sched, _ = {fn}(p, m)
    assert len(sched) == p, f'p={p}, m={m}: expected {p} stages'
    for s, ops in enumerate(sched):
        assert len(ops) == 2 * m, f'p={p}, m={m}, stage {s}: expected {2 * m} ops, got {len(ops)}'
        for i in range(m):
            assert ops.count(f'F{i}') == 1 and ops.count(f'B{i}') == 1, f'p={p}, m={m}, stage {s}: micro-batch {i} not run exactly once per direction'
            assert ops.index(f'F{i}') < ops.index(f'B{i}'), f'p={p}, m={m}, stage {s}: B{i} before F{i}'
""",
        },
        {
            "name": "Peak in-flight micro-batches is min(p - s, m)",
            "code": """
import torch
for p, m in [(4, 8), (5, 6), (4, 2), (3, 3), (1, 5), (6, 10)]:
    sched, _ = {fn}(p, m)
    for s, ops in enumerate(sched):
        in_flight, peak = 0, 0
        for op in ops:
            in_flight += 1 if op[0] == 'F' else -1
            assert in_flight >= 0, f'p={p}, m={m}, stage {s}: backward before its forward'
            peak = max(peak, in_flight)
        expected = min(p - s, m)
        assert peak == expected, f'p={p}, m={m}, stage {s}: peak in-flight {peak}, expected {expected}'
""",
        },
        {
            "name": "m < p: warmup is capped at m",
            "code": """
import torch
sched, _ = {fn}(4, 2)
assert sched[0] == ['F0', 'F1', 'B0', 'B1'], f'Stage 0: got {sched[0]}'
assert sched[1] == ['F0', 'F1', 'B0', 'B1'], f'Stage 1: got {sched[1]}'
assert sched[2] == ['F0', 'F1', 'B0', 'B1'], f'Stage 2: got {sched[2]}'
assert sched[3] == ['F0', 'B0', 'F1', 'B1'], f'Stage 3: got {sched[3]}'
""",
        },
        {
            "name": "Single stage is F0 B0 F1 B1 ...",
            "code": """
import torch
sched, bubble = {fn}(1, 3)
assert sched == [['F0', 'B0', 'F1', 'B1', 'F2', 'B2']], f'Got {sched}'
assert abs(bubble) < 1e-12, f'Single stage has no bubble, got {bubble}'
""",
        },
        {
            "name": "Bubble fraction values",
            "code": """
import torch
import math
cases = [((4, 8), 3 / 11), ((2, 2), 1 / 3), ((1, 5), 0.0), ((4, 2), 3 / 5), ((8, 32), 7 / 39)]
for (p, m), expected in cases:
    _, bubble = {fn}(p, m)
    assert isinstance(bubble, float), f'p={p}, m={m}: bubble must be a float, got {type(bubble)}'
    assert math.isclose(bubble, expected, rel_tol=1e-9), f'p={p}, m={m}: bubble {bubble}, expected {expected}'
""",
        },
        {
            "name": "Schedule never deadlocks (unit-time simulation)",
            "code": """
import torch
def simulate(schedule, p):
    # One op per stage per tick; an op may start only after its cross-stage dependency
    # finished in an earlier tick: F{i} on s needs F{i} on s-1, B{i} on s needs B{i} on s+1.
    ptr = [0] * p
    done = set()
    total = sum(len(ops) for ops in schedule)
    finished, ticks = 0, 0
    while finished < total:
        ready = []
        for s in range(p):
            if ptr[s] == len(schedule[s]):
                continue
            op = schedule[s][ptr[s]]
            kind, i = op[0], int(op[1:])
            if kind == 'F':
                ok = s == 0 or ('F', s - 1, i) in done
            else:
                ok = s == p - 1 or ('B', s + 1, i) in done
            if ok:
                ready.append((s, kind, i))
        assert ready, f'Deadlock after {ticks} ticks with {finished}/{total} ops done'
        for s, kind, i in ready:
            done.add((kind, s, i))
            ptr[s] += 1
            finished += 1
        ticks += 1
    return ticks

for p, m in [(1, 1), (2, 2), (4, 2), (4, 8), (5, 3), (3, 9), (6, 6)]:
    sched, bubble = {fn}(p, m)
    ticks = simulate(sched, p)
    assert ticks == 2 * (m + p - 1), f'p={p}, m={m}: makespan {ticks}, expected {2 * (m + p - 1)}'
    busy_fraction = 1 - 2 * m / ticks
    assert abs(busy_fraction - bubble) < 1e-9, f'p={p}, m={m}: simulated bubble {busy_fraction} != {bubble}'
""",
        },
    ],
    "solution": '''def pipeline_1f1b_schedule(num_stages, num_microbatches):
    p, m = num_stages, num_microbatches
    schedule = []
    for s in range(p):
        # Warmup: stage s runs (p - s - 1) forwards before its first backward
        warmup = min(p - s - 1, m)
        ops = [f"F{i}" for i in range(warmup)]

        # Steady state: one forward, then one backward, per step
        for j in range(m - warmup):
            ops.append(f"F{warmup + j}")
            ops.append(f"B{j}")

        # Cooldown: the remaining backwards drain the pipeline
        ops += [f"B{j}" for j in range(m - warmup, m)]
        schedule.append(ops)

    # Ideal bubble with equal forward/backward times: (p - 1) / (m + p - 1)
    bubble_fraction = (p - 1) / (m + p - 1)
    return schedule, bubble_fraction''',
    "demo": """p, m = 4, 8
schedule, bubble = pipeline_1f1b_schedule(p, m)

for s, ops in enumerate(schedule):
    print(f"stage {s}: {' '.join(ops)}")

for s in range(p):
    in_flight, peak = 0, 0
    for op in schedule[s]:
        in_flight += 1 if op[0] == "F" else -1
        peak = max(peak, in_flight)
    print(f"stage {s}: peak in-flight = {peak} (GPipe would hold {m})")  # expect min(p - s, m)

print(f"bubble fraction = {bubble:.4f}")  # expect 3/11 = 0.2727""",
}
