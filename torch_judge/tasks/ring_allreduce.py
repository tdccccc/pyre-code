"""Ring All-Reduce task."""

TASK = {
    "title": "Ring All-Reduce",
    "title_zh": "环形 All-Reduce",
    "difficulty": "Medium",
    "description_en": "Implement ring all-reduce (single-process simulation of the bandwidth-optimal collective used by NCCL).\n\nEvery data-parallel rank holds a full gradient buffer and needs the elementwise sum of all ranks' buffers. A naive reduce-to-one-rank sends the whole buffer over one link per step. Ring all-reduce instead splits each buffer into W chunks and circulates them around a ring of W ranks in two phases: a **reduce-scatter** (each rank ends up owning one fully reduced chunk) and an **all-gather** (the reduced chunks travel around the ring until every rank has all of them). Each step a rank sends one chunk of about L/W elements, and there are 2(W-1) steps, so each rank sends 2(W-1)/W x L elements in total. That is always below 2L no matter how many ranks there are, so per-rank bandwidth does not grow with W. This is why the algorithm is bandwidth optimal.\n\n**Signature:** `ring_allreduce(tensors) -> tuple[list[Tensor], list[list[tuple[int, int, int]]]]`\n\n**Parameters:**\n- `tensors` — list of `W >= 1` one-dimensional tensors of the same length `L >= W`. `tensors[r]` is the buffer held by rank `r`. Do not modify them.\n\n**Returns:** `(outputs, log)`\n- `outputs` — list of `W` tensors. Each equals the elementwise sum of all inputs, has the same dtype as the inputs, and is an independent tensor (no shared storage).\n- `log` — one entry per communication step, in execution order: the `W-1` reduce-scatter steps first, then the `W-1` all-gather steps (`2(W-1)` entries). Each entry is a list of `W` tuples `(src, dst, chunk_index)`, one per rank, sorted by `src`, with `dst = (src + 1) % W`.\n\n**Algorithm:**\n1. Split every buffer into `W` chunks exactly as `torch.tensor_split(buf, W)` does. Chunk `c` of rank `r` is `chunks[r][c]`.\n2. Reduce-scatter, for steps `t = 0 .. W-2`: every rank `r` sends chunk `(r - t) mod W` to rank `(r + 1) mod W`, which **adds** it into its own copy of that chunk. All sends in one step use the values held *before* that step. Afterwards rank `r` holds the fully reduced chunk `(r + 1) mod W`.\n3. All-gather, for steps `t = 0 .. W-2`: every rank `r` sends chunk `(r + 1 - t) mod W` to rank `(r + 1) mod W`, which **overwrites** its copy of that chunk.\n4. `outputs[r]` is the concatenation of rank `r`'s `W` chunks.\n\n**Constraints:**\n- For `W = 1` there are no communication steps: `log == []` and `outputs[0]` is a copy of the input.\n- Integer tensors must be summed exactly, with no conversion to float.\n- Do not mutate the input tensors.",
    "description_zh": "实现环形 All-Reduce（单进程模拟 NCCL 使用的带宽最优集合通信）。\n\n每个数据并行 rank 持有一份完整的梯度缓冲区，需要得到所有 rank 缓冲区的逐元素求和。朴素的 reduce-to-one 每一步都要把整个缓冲区通过一条链路发送。环形 All-Reduce 则把每个缓冲区切成 W 个分块，在由 W 个 rank 组成的环上分两个阶段传递：**reduce-scatter**（每个 rank 最终拥有一个完全归约好的分块）和 **all-gather**（归约后的分块在环上传递，直到每个 rank 都拿到全部分块）。每一步 rank 发送一个约 L/W 个元素的分块，共 2(W-1) 步，因此每个 rank 总共发送 2(W-1)/W x L 个元素。无论有多少个 rank，这个值都小于 2L，所以单 rank 的带宽开销不随 W 增长，这就是该算法带宽最优的原因。\n\n**签名:** `ring_allreduce(tensors) -> tuple[list[Tensor], list[list[tuple[int, int, int]]]]`\n\n**参数:**\n- `tensors` — 长度为 `W >= 1` 的列表，每个元素是长度相同的一维张量，长度 `L >= W`。`tensors[r]` 是 rank `r` 持有的缓冲区。不得修改它们。\n\n**返回:** `(outputs, log)`\n- `outputs` — 包含 `W` 个张量的列表。每个张量都等于所有输入的逐元素求和，dtype 与输入相同，并且是独立的张量（不共享存储）。\n- `log` — 按执行顺序每个通信步骤一条记录：先是 `W-1` 个 reduce-scatter 步，然后是 `W-1` 个 all-gather 步（共 `2(W-1)` 条）。每条记录是 `W` 个元组 `(src, dst, chunk_index)` 的列表，每个 rank 一个，按 `src` 排序，且 `dst = (src + 1) % W`。\n\n**算法:**\n1. 将每个缓冲区切成 `W` 个分块，切法与 `torch.tensor_split(buf, W)` 完全一致。rank `r` 的第 `c` 块为 `chunks[r][c]`。\n2. Reduce-scatter，步数 `t = 0 .. W-2`：每个 rank `r` 把分块 `(r - t) mod W` 发送给 rank `(r + 1) mod W`，后者将其**累加**到自己对应分块的副本上。同一步中的所有发送都使用该步之前的数值。结束后，rank `r` 持有完全归约好的分块 `(r + 1) mod W`。\n3. All-gather，步数 `t = 0 .. W-2`：每个 rank `r` 把分块 `(r + 1 - t) mod W` 发送给 rank `(r + 1) mod W`，后者**覆盖**自己对应分块的副本。\n4. `outputs[r]` 是 rank `r` 的 `W` 个分块拼接而成的结果。\n\n**约束:**\n- 当 `W = 1` 时没有通信步骤：`log == []`，且 `outputs[0]` 是输入的副本。\n- 整数张量必须精确求和，不得转换为浮点数。\n- 不得修改输入张量。",
    "function_name": "ring_allreduce",
    "hint": "chunks[r][c] = clone of tensors[r].tensor_split(W)[c]   (keep copies, don't touch inputs)\nfor t in range(W-1):          # reduce-scatter\n    sent = [chunks[r][(r-t)%W] for r in range(W)]        # snapshot before the step\n    for r: chunks[(r+1)%W][(r-t)%W] += sent[r];  log.append((r, (r+1)%W, (r-t)%W))\nfor t in range(W-1):          # all-gather\n    sent = [chunks[r][(r+1-t)%W] ...];  chunks[(r+1)%W][(r+1-t)%W] = sent[r]\noutputs[r] = torch.cat(chunks[r])",
    "hint_zh": "chunks[r][c] = tensors[r].tensor_split(W)[c] 的副本（保留副本，不要修改输入）\nfor t in range(W-1):          # reduce-scatter\n    sent = [chunks[r][(r-t)%W] for r in range(W)]        # 本步之前的快照\n    for r: chunks[(r+1)%W][(r-t)%W] += sent[r];  log.append((r, (r+1)%W, (r-t)%W))\nfor t in range(W-1):          # all-gather\n    sent = [chunks[r][(r+1-t)%W] ...];  chunks[(r+1)%W][(r+1-t)%W] = sent[r]\noutputs[r] = torch.cat(chunks[r])",
    "tests": [
        {
            "name": "Output count and shapes",
            "code": """
import torch
torch.manual_seed(0)
W, L = 3, 7
tensors = [torch.randn(L) for _ in range(W)]
outputs, log = {fn}(tensors)
assert isinstance(outputs, list) and len(outputs) == W, f'Expected {W} outputs, got {len(outputs)}'
assert all(o.shape == (L,) for o in outputs), f'Each output should have shape ({L},)'
assert len(log) == 2 * (W - 1), f'Expected {2 * (W - 1)} log entries, got {len(log)}'
""",
        },
        {
            "name": "Sums are correct for W=1..5 (L not divisible by W)",
            "code": """
import torch
torch.manual_seed(1)
L = 13  # 13 is not divisible by 2, 3, 4 or 5
for W in [1, 2, 3, 4, 5]:
    tensors = [torch.randn(L, dtype=torch.float64) for _ in range(W)]
    ref = tensors[0].clone()
    for t in tensors[1:]:
        ref = ref + t
    outputs, _ = {fn}(tensors)
    assert len(outputs) == W, f'W={W}: expected {W} outputs'
    for r, o in enumerate(outputs):
        assert o.shape == (L,), f'W={W} rank {r}: shape {tuple(o.shape)}'
        assert torch.allclose(o, ref, atol=1e-10), f'W={W} rank {r}: output is not the elementwise sum'
""",
        },
        {
            "name": "Exact log for W=3",
            "code": """
import torch
torch.manual_seed(2)
tensors = [torch.randn(6) for _ in range(3)]
_, log = {fn}(tensors)
expected = [
    [(0, 1, 0), (1, 2, 1), (2, 0, 2)],  # reduce-scatter t=0: chunk r
    [(0, 1, 2), (1, 2, 0), (2, 0, 1)],  # reduce-scatter t=1: chunk (r-1) mod 3
    [(0, 1, 1), (1, 2, 2), (2, 0, 0)],  # all-gather t=0: chunk (r+1) mod 3
    [(0, 1, 0), (1, 2, 1), (2, 0, 2)],  # all-gather t=1: chunk r
]
assert log == expected, f'Log mismatch:\\n got      {log}\\n expected {expected}'
""",
        },
        {
            "name": "Log structure for W=4",
            "code": """
import torch
torch.manual_seed(3)
W, L = 4, 10
tensors = [torch.randn(L) for _ in range(W)]
_, log = {fn}(tensors)
assert len(log) == 2 * (W - 1), f'Expected {2 * (W - 1)} steps, got {len(log)}'
for step, entry in enumerate(log):
    assert len(entry) == W, f'Step {step}: expected {W} transfers, got {len(entry)}'
    assert [src for src, _, _ in entry] == list(range(W)), f'Step {step}: entries must be sorted by src, one per rank'
    for src, dst, c in entry:
        assert dst == (src + 1) % W, f'Step {step}: rank {src} must send to {(src + 1) % W}, not {dst}'
    chunks = [c for _, _, c in entry]
    if step < W - 1:
        t = step
        expected = [(r - t) % W for r in range(W)]
    else:
        t = step - (W - 1)
        expected = [(r + 1 - t) % W for r in range(W)]
    assert chunks == expected, f'Step {step}: chunk indices {chunks}, expected {expected}'
""",
        },
        {
            "name": "Bandwidth is 2(W-1)/W of the buffer per rank",
            "code": """
import torch
torch.manual_seed(4)
for W, L in [(4, 16), (3, 18)]:
    tensors = [torch.randn(L) for _ in range(W)]
    _, log = {fn}(tensors)
    chunk_len = [c.numel() for c in torch.tensor_split(torch.empty(L), W)]
    sent = [0] * W
    for entry in log:
        for src, _, c in entry:
            sent[src] += chunk_len[c]
    expected = 2 * (W - 1) * L / W
    for r in range(W):
        assert sent[r] == expected, f'W={W}: rank {r} sent {sent[r]} elements, expected {expected}'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(5)
W, L = 4, 10
tensors = [torch.randn(L) for _ in range(W)]
before = [t.clone() for t in tensors]
{fn}(tensors)
for r in range(W):
    assert torch.equal(tensors[r], before[r]), f'Input tensor {r} was modified'
""",
        },
        {
            "name": "Outputs are independent tensors",
            "code": """
import torch
torch.manual_seed(6)
W, L = 3, 6
tensors = [torch.randn(L) for _ in range(W)]
ref = sum(tensors[1:], tensors[0].clone())
outputs, _ = {fn}(tensors)
for i in range(W):
    for j in range(i + 1, W):
        assert outputs[i].data_ptr() != outputs[j].data_ptr(), f'Outputs {i} and {j} share storage'
outputs[0].add_(1000.0)
for r in range(1, W):
    assert torch.allclose(outputs[r], ref, atol=1e-5), f'Modifying output 0 changed output {r}'
""",
        },
        {
            "name": "Integer tensors sum exactly",
            "code": """
import torch
g = torch.Generator().manual_seed(7)
W, L = 4, 9
tensors = [torch.randint(-10**12, 10**12, (L,), generator=g, dtype=torch.int64) for _ in range(W)]
ref = tensors[0].clone()
for t in tensors[1:]:
    ref = ref + t
outputs, _ = {fn}(tensors)
for r, o in enumerate(outputs):
    assert o.dtype == torch.int64, f'Rank {r}: dtype {o.dtype}, expected torch.int64'
    assert torch.equal(o, ref), f'Rank {r}: integer sum is not exact'
""",
        },
    ],
    "solution": '''def ring_allreduce(tensors):
    W = len(tensors)
    # chunks[r][c]: chunk c of rank r's buffer (copies, so the inputs are never touched)
    chunks = [[c.clone() for c in t.tensor_split(W)] for t in tensors]
    log = []

    # Phase 1: reduce-scatter. Rank r sends chunk (r - t) to rank r+1, which adds it in.
    for t in range(W - 1):
        sent = [chunks[r][(r - t) % W] for r in range(W)]  # values from before this step
        entry = []
        for r in range(W):
            dst, c = (r + 1) % W, (r - t) % W
            chunks[dst][c] = chunks[dst][c] + sent[r]
            entry.append((r, dst, c))
        log.append(entry)

    # Phase 2: all-gather. Rank r forwards chunk (r + 1 - t); the receiver overwrites it.
    for t in range(W - 1):
        sent = [chunks[r][(r + 1 - t) % W] for r in range(W)]
        entry = []
        for r in range(W):
            dst, c = (r + 1) % W, (r + 1 - t) % W
            chunks[dst][c] = sent[r]
            entry.append((r, dst, c))
        log.append(entry)

    # Every rank now holds all reduced chunks; concatenation copies into fresh tensors
    outputs = [torch.cat(chunks[r]) for r in range(W)]
    return outputs, log''',
    "demo": """torch.manual_seed(0)
W, L = 4, 10
tensors = [torch.randn(L) for _ in range(W)]

outputs, log = ring_allreduce(tensors)
ref = torch.stack(tensors).sum(dim=0)

print("steps:", len(log), "| transfers per step:", len(log[0]))  # expect 6 | 4
print("max error vs elementwise sum:", max((o - ref).abs().max().item() for o in outputs))  # expect ~0
print("reduce-scatter step 0:", log[0])  # [(0, 1, 0), (1, 2, 1), (2, 3, 2), (3, 0, 3)]
print("all-gather step 3:", log[-1])""",
}
