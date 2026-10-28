"""Decode Roofline & KV Cache task."""

TASK = {
    "title": "Decode Roofline & KV Cache",
    "title_zh": "解码 Roofline 与 KV Cache",
    "difficulty": "Medium",
    "description_en": "Estimate the performance of one autoregressive decode step for LLM serving, using the roofline model. Decode is usually limited by memory bandwidth, not by FLOPs, so the KV cache size and the batch size set the throughput. This is the reasoning behind GQA, paged KV caches and continuous batching.\n\n**Signature:** `inference_roofline(num_params, n_layers, d_model, n_heads, n_kv_heads, seq_len, batch_size, peak_flops, mem_bandwidth, bytes_per_param=2) -> dict`\n\n**Parameters:**\n- `num_params` — total model parameters\n- `n_layers`, `d_model`, `n_heads`, `n_kv_heads` — architecture; `head_dim = d_model // n_heads`; `n_kv_heads` divides `n_heads` (grouped-query attention; `n_kv_heads == n_heads` is multi-head attention)\n- `seq_len` — tokens already in each sequence's KV cache\n- `batch_size` — sequences decoded together; each one generates one new token in this step\n- `peak_flops` — device peak FLOP/s; `mem_bandwidth` — bytes/s\n- `bytes_per_param` — bytes per weight and per KV element (2 for bf16/fp16). The KV cache uses the same dtype as the weights.\n\n**Model of one decode step:** each of the `batch_size` sequences produces one token while attending over its `seq_len` cached tokens. Weights are read once for the whole batch. Ignore attention FLOPs, activations and the softmax. Count only weight and KV-cache traffic.\n\n**Returns:** a dict with\n- `kv_cache_bytes` (int)\n- `flops` (int)\n- `bytes` (int)\n- `arithmetic_intensity` (float, FLOP per byte)\n- `bound` (str): `memory` or `compute`\n- `step_time` (float, seconds)\n- `tokens_per_sec` (float)\n- `mfu` (float)\n\n**Formulas:**\n```\nhead_dim             = d_model // n_heads\nkv_cache_bytes       = 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_param   (2 = K and V)\nflops                = 2 * num_params * batch_size\nbytes                = num_params * bytes_per_param + kv_cache_bytes   (weights read once, whole KV cache read once)\narithmetic_intensity = flops / bytes\nridge                = peak_flops / mem_bandwidth\nbound                = memory if arithmetic_intensity < ridge else compute\nstep_time            = max(flops / peak_flops, bytes / mem_bandwidth)\ntokens_per_sec       = batch_size / step_time\nmfu                  = flops / step_time / peak_flops\n```\n\n**Worked example:** LLaMA-7B-like model (`num_params=6738415616`, `n_layers=32`, `d_model=4096`, 32 heads with 32 KV heads), `seq_len=2048`, `batch_size=1`, A100-like device (`peak_flops=312e12`, `mem_bandwidth=2.0e12`), bf16:\n- `kv_cache_bytes = 2*32*1*2048*32*128*2 = 1073741824`\n- `flops = 2 * 6738415616 = 13476831232`; `bytes = 13476831232 + 1073741824 = 14550573056`\n- `arithmetic_intensity` = 0.926 FLOP/byte; ridge = 156, so `bound = memory`\n- `step_time = bytes / mem_bandwidth` = 7.28 ms; `tokens_per_sec` = 137.5; `mfu` = 0.0059\n\n**Why GQA and larger batches help:** decode reads all weights and the whole KV cache for only a few FLOPs per byte. GQA shrinks the KV cache because fewer KV heads means fewer bytes per cached token, which raises arithmetic intensity. Larger batches amortize the single weight read over more tokens, so throughput rises until the step becomes compute bound (then `mfu` = 1 exactly).",
    "description_zh": "使用 roofline 模型估算 LLM 推理中一次自回归解码步的性能。解码通常受显存带宽限制而非 FLOPs 限制，因此 KV Cache 的大小和 batch size 决定了吞吐。这正是 GQA、分页 KV Cache 和连续批处理（continuous batching）的设计依据。\n\n**签名:** `inference_roofline(num_params, n_layers, d_model, n_heads, n_kv_heads, seq_len, batch_size, peak_flops, mem_bandwidth, bytes_per_param=2) -> dict`\n\n**参数:**\n- `num_params` — 模型总参数量\n- `n_layers`、`d_model`、`n_heads`、`n_kv_heads` — 模型结构；`head_dim = d_model // n_heads`；`n_kv_heads` 整除 `n_heads`（分组查询注意力 GQA；`n_kv_heads == n_heads` 即多头注意力 MHA）\n- `seq_len` — 每个序列的 KV Cache 中已有的 token 数\n- `batch_size` — 同时解码的序列数；本步中每个序列各生成一个新 token\n- `peak_flops` — 设备峰值 FLOP/s；`mem_bandwidth` — 显存带宽（字节/秒）\n- `bytes_per_param` — 每个权重和每个 KV 元素的字节数（bf16/fp16 为 2）。KV Cache 与权重使用相同的数据类型。\n\n**一次解码步的模型:** `batch_size` 个序列中的每一个都基于自己 `seq_len` 个已缓存的 token 生成一个新 token。整个 batch 只读取一次权重。忽略注意力的 FLOPs、激活值和 softmax，只统计权重和 KV Cache 的访存量。\n\n**返回:** 字典，包含\n- `kv_cache_bytes`（int）\n- `flops`（int）\n- `bytes`（int）\n- `arithmetic_intensity`（float，每字节的 FLOP 数）\n- `bound`（str）：`memory` 或 `compute`\n- `step_time`（float，秒）\n- `tokens_per_sec`（float）\n- `mfu`（float）\n\n**公式:**\n```\nhead_dim             = d_model // n_heads\nkv_cache_bytes       = 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_param   （2 = K 和 V）\nflops                = 2 * num_params * batch_size\nbytes                = num_params * bytes_per_param + kv_cache_bytes   （权重读一次，整个 KV Cache 读一次）\narithmetic_intensity = flops / bytes\nridge                = peak_flops / mem_bandwidth\nbound                = memory 若 arithmetic_intensity < ridge，否则 compute\nstep_time            = max(flops / peak_flops, bytes / mem_bandwidth)\ntokens_per_sec       = batch_size / step_time\nmfu                  = flops / step_time / peak_flops\n```\n\n**手算示例:** LLaMA-7B 类模型（`num_params=6738415616`，`n_layers=32`，`d_model=4096`，32 个头且 32 个 KV 头），`seq_len=2048`，`batch_size=1`，A100 类设备（`peak_flops=312e12`，`mem_bandwidth=2.0e12`），bf16：\n- `kv_cache_bytes = 2*32*1*2048*32*128*2 = 1073741824`\n- `flops = 2 * 6738415616 = 13476831232`；`bytes = 13476831232 + 1073741824 = 14550573056`\n- `arithmetic_intensity` = 0.926 FLOP/字节；ridge = 156，因此 `bound = memory`\n- `step_time = bytes / mem_bandwidth` = 7.28 ms；`tokens_per_sec` = 137.5；`mfu` = 0.0059\n\n**GQA 与更大 batch 为何有效:** 解码每个字节只做极少的 FLOPs，却要读取全部权重和整个 KV Cache。GQA 减少了 KV 头数，从而减少每个缓存 token 的字节数，提高算术强度。更大的 batch 把一次权重读取摊到更多 token 上，吞吐随之上升，直到该步变为 compute bound（此时 `mfu` 恰好为 1）。",
    "function_name": "inference_roofline",
    "hint": "head_dim = d_model // n_heads\nkv = 2 * n_layers * batch * seq_len * n_kv_heads * head_dim * bytes_per_param\nflops = 2 * num_params * batch;  bytes = num_params * bytes_per_param + kv\nai = flops / bytes;  ridge = peak_flops / mem_bandwidth\nbound = \"memory\" if ai < ridge else \"compute\"\nstep = max(flops / peak_flops, bytes / mem_bandwidth)\ntokens_per_sec = batch / step;  mfu = flops / step / peak_flops",
    "hint_zh": "head_dim = d_model // n_heads\nkv = 2 * n_layers * batch * seq_len * n_kv_heads * head_dim * bytes_per_param\nflops = 2 * num_params * batch;  bytes = num_params * bytes_per_param + kv\nai = flops / bytes;  ridge = peak_flops / mem_bandwidth\nbound = \"memory\" if ai < ridge else \"compute\"\nstep = max(flops / peak_flops, bytes / mem_bandwidth)\ntokens_per_sec = batch / step;  mfu = flops / step / peak_flops",
    "tests": [
        {
            "name": "Return types",
            "code": """
import torch
out = {fn}(6738415616, 32, 4096, 32, 32, 2048, 1, 312e12, 2.0e12)
for k in ("kv_cache_bytes", "flops", "bytes"):
    assert type(out[k]) is int, f'{k} should be int, got {type(out[k]).__name__}'
for k in ("arithmetic_intensity", "step_time", "tokens_per_sec", "mfu"):
    assert type(out[k]) is float, f'{k} should be float, got {type(out[k]).__name__}'
assert out["bound"] in ("memory", "compute"), f'bound should be memory or compute, got {out["bound"]}'
""",
        },
        {
            "name": "LLaMA-7B on A100, batch 1 (hand-computed)",
            "code": """
import torch
import math
P = 6738415616  # LLaMA-7B parameter count
out = {fn}(P, 32, 4096, 32, 32, 2048, 1, 312e12, 2.0e12)
# head_dim = 128; kv = 2 * 32 * 1 * 2048 * 32 * 128 * 2 = 2^30
assert out["kv_cache_bytes"] == 1073741824, f'kv_cache_bytes: expected 1073741824, got {out["kv_cache_bytes"]}'
assert out["flops"] == 13476831232, f'flops: expected 13476831232 (2P), got {out["flops"]}'
assert out["bytes"] == 14550573056, f'bytes: expected 14550573056 (2P + 2^30), got {out["bytes"]}'
ai = 13476831232 / 14550573056
assert math.isclose(out["arithmetic_intensity"], ai, rel_tol=1e-9), f'arithmetic_intensity: expected {ai}, got {out["arithmetic_intensity"]}'
# ridge = 312e12 / 2e12 = 156 FLOP/byte; AI ~ 0.93 is far below it
assert out["bound"] == "memory", f'batch 1 decode should be memory bound, got {out["bound"]}'
assert math.isclose(out["step_time"], 0.007275286528, rel_tol=1e-9), f'step_time: expected 0.007275286528 (bytes / 2e12), got {out["step_time"]}'
assert math.isclose(out["tokens_per_sec"], 1 / 0.007275286528, rel_tol=1e-9), f'tokens_per_sec: got {out["tokens_per_sec"]}'
assert math.isclose(out["mfu"], ai / 156.0, rel_tol=1e-9), f'mfu should be AI / ridge when memory bound, got {out["mfu"]}'
""",
        },
        {
            "name": "Huge batch becomes compute bound",
            "code": """
import torch
import math
P = 6738415616
# batch 2048 at seq_len 128: kv = 2048 * 2*32*128*32*128*2 = 137438953472 bytes
# AI = 2P*2048 / (2P + kv) ~ 182.9 > ridge 156
out = {fn}(P, 32, 4096, 32, 32, 128, 2048, 312e12, 2.0e12)
assert out["bound"] == "compute", f'large batch decode should be compute bound, got {out["bound"]}'
assert math.isclose(out["step_time"], out["flops"] / 312e12, rel_tol=1e-9), 'compute bound: step_time should equal flops / peak_flops'
assert math.isclose(out["mfu"], 1.0, rel_tol=1e-9), f'compute bound: mfu should be 1, got {out["mfu"]}'
""",
        },
        {
            "name": "GQA (8 KV heads) shrinks the KV cache 4x",
            "code": """
import torch
P = 6738415616
mha = {fn}(P, 32, 4096, 32, 32, 2048, 1, 312e12, 2.0e12)
gqa = {fn}(P, 32, 4096, 32, 8, 2048, 1, 312e12, 2.0e12)
assert gqa["kv_cache_bytes"] == 268435456, f'GQA kv_cache_bytes: expected 268435456, got {gqa["kv_cache_bytes"]}'
assert mha["kv_cache_bytes"] == 4 * gqa["kv_cache_bytes"], 'MHA KV cache should be 4x the 8-KV-head GQA cache'
assert gqa["bytes"] == 2 * P + 268435456, f'GQA bytes: expected {2 * P + 268435456}, got {gqa["bytes"]}'
assert gqa["arithmetic_intensity"] > mha["arithmetic_intensity"], 'smaller KV cache should raise arithmetic intensity'
""",
        },
        {
            "name": "bytes_per_param=1 halves weight and KV bytes",
            "code": """
import torch
P = 6738415616
fp16 = {fn}(P, 32, 4096, 32, 32, 2048, 1, 312e12, 2.0e12, bytes_per_param=2)
int8 = {fn}(P, 32, 4096, 32, 32, 2048, 1, 312e12, 2.0e12, bytes_per_param=1)
assert int8["kv_cache_bytes"] * 2 == fp16["kv_cache_bytes"], 'KV cache should halve with 1 byte per element'
assert int8["bytes"] - int8["kv_cache_bytes"] == P, f'1-byte weights should take P bytes, got {int8["bytes"] - int8["kv_cache_bytes"]}'
assert fp16["bytes"] - fp16["kv_cache_bytes"] == 2 * P, 'bf16 weights should take 2P bytes'
assert int8["flops"] == fp16["flops"], 'dtype must not change FLOPs'
assert int8["arithmetic_intensity"] > fp16["arithmetic_intensity"], 'fewer bytes per step should raise arithmetic intensity'
""",
        },
        {
            "name": "Batching amortizes the weight read",
            "code": """
import torch
import math
P = 6738415616
# batch 2: flops 4P, bytes = 2P + 2*2^30 = 15624314880, step = 0.00781215744 s
out = {fn}(P, 32, 4096, 32, 32, 2048, 2, 312e12, 2.0e12)
assert out["bound"] == "memory", f'batch 2 at 2048 context should still be memory bound, got {out["bound"]}'
assert math.isclose(out["step_time"], 0.00781215744, rel_tol=1e-9), f'batch 2 step_time: expected 0.00781215744, got {out["step_time"]}'
assert math.isclose(out["tokens_per_sec"], 2 / 0.00781215744, rel_tol=1e-9), f'batch 2 tokens_per_sec: got {out["tokens_per_sec"]}'
batch1 = {fn}(P, 32, 4096, 32, 32, 2048, 1, 312e12, 2.0e12)
assert out["tokens_per_sec"] > 1.8 * batch1["tokens_per_sec"], 'batching should nearly double throughput while memory bound'
""",
        },
        {
            "name": "mfu = min(1, AI / ridge) across batch sizes",
            "code": """
import torch
import math
P = 6738415616
ridge = 312e12 / 2.0e12
seen = set()
for B in (1, 4, 64, 512, 2048):
    out = {fn}(P, 32, 4096, 32, 8, 512, B, 312e12, 2.0e12)
    seen.add(out["bound"])
    assert 0 < out["mfu"] <= 1.0 + 1e-12, f'batch {B}: mfu must be in (0, 1], got {out["mfu"]}'
    expected_mfu = min(1.0, out["arithmetic_intensity"] / ridge)
    assert math.isclose(out["mfu"], expected_mfu, rel_tol=1e-9), f'batch {B}: mfu {out["mfu"]} != min(1, AI/ridge) = {expected_mfu}'
    assert (out["bound"] == "compute") == (out["arithmetic_intensity"] >= ridge), f'batch {B}: bound rule wrong for AI={out["arithmetic_intensity"]}'
    assert math.isclose(out["step_time"], max(out["flops"] / 312e12, out["bytes"] / 2.0e12), rel_tol=1e-9), f'batch {B}: step_time must be the max of the two limits'
    assert math.isclose(out["tokens_per_sec"], B / out["step_time"], rel_tol=1e-9), f'batch {B}: tokens_per_sec must be batch / step_time'
assert seen == {"memory", "compute"}, f'sweep should cover both regimes, saw {seen}'
""",
        },
        {
            "name": "Random configs match reference formula",
            "code": """
import torch
import math
torch.manual_seed(5)
for _ in range(5):
    L = int(torch.randint(4, 81, (1,)))
    heads = 8 * int(torch.randint(1, 9, (1,)))
    d = heads * 64 * int(torch.randint(1, 3, (1,)))
    kvh = heads // (2 ** int(torch.randint(0, 4, (1,))))
    P = int(torch.randint(10**9, 10**11, (1,)))
    s = int(torch.randint(64, 8193, (1,)))
    B = int(torch.randint(1, 513, (1,)))
    bpp = int(torch.randint(1, 3, (1,)))
    peak = float(int(torch.randint(100, 1000, (1,)))) * 1e12
    bw = float(int(torch.randint(1, 40, (1,)))) * 1e11
    out = {fn}(P, L, d, heads, kvh, s, B, peak, bw, bytes_per_param=bpp)
    hd = d // heads
    kv_ref = 2 * L * B * s * kvh * hd * bpp
    fl_ref = 2 * P * B
    by_ref = P * bpp + kv_ref
    ai_ref = fl_ref / by_ref
    bound_ref = "memory" if ai_ref < peak / bw else "compute"
    st_ref = max(fl_ref / peak, by_ref / bw)
    cfg = f'P={P} L={L} d={d} heads={heads} kv_heads={kvh} s={s} B={B} bpp={bpp} peak={peak} bw={bw}'
    assert out["kv_cache_bytes"] == kv_ref, f'{cfg}: kv_cache_bytes {out["kv_cache_bytes"]} != {kv_ref}'
    assert out["flops"] == fl_ref, f'{cfg}: flops {out["flops"]} != {fl_ref}'
    assert out["bytes"] == by_ref, f'{cfg}: bytes {out["bytes"]} != {by_ref}'
    assert math.isclose(out["arithmetic_intensity"], ai_ref, rel_tol=1e-9), f'{cfg}: AI mismatch'
    assert out["bound"] == bound_ref, f'{cfg}: bound {out["bound"]} != {bound_ref}'
    assert math.isclose(out["step_time"], st_ref, rel_tol=1e-9), f'{cfg}: step_time {out["step_time"]} != {st_ref}'
    assert math.isclose(out["tokens_per_sec"], B / st_ref, rel_tol=1e-9), f'{cfg}: tokens_per_sec mismatch'
    assert math.isclose(out["mfu"], fl_ref / st_ref / peak, rel_tol=1e-9), f'{cfg}: mfu mismatch'
""",
        },
    ],
    "solution": '''def inference_roofline(num_params, n_layers, d_model, n_heads, n_kv_heads, seq_len, batch_size, peak_flops, mem_bandwidth, bytes_per_param=2):
    head_dim = d_model // n_heads

    # KV cache: K and V (factor 2) for every layer, every cached token, every KV head
    kv_cache_bytes = 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * bytes_per_param

    # One decode step: 2 FLOPs per parameter per token; weights are read once for the whole batch
    flops = 2 * num_params * batch_size
    total_bytes = num_params * bytes_per_param + kv_cache_bytes

    intensity = flops / total_bytes
    ridge = peak_flops / mem_bandwidth  # FLOP/byte where compute and memory time are equal
    bound = "memory" if intensity < ridge else "compute"

    # The slower of the two limits sets the step time (roofline)
    step_time = max(flops / peak_flops, total_bytes / mem_bandwidth)
    tokens_per_sec = batch_size / step_time
    mfu = flops / step_time / peak_flops

    return {
        "kv_cache_bytes": kv_cache_bytes,
        "flops": flops,
        "bytes": total_bytes,
        "arithmetic_intensity": intensity,
        "bound": bound,
        "step_time": step_time,
        "tokens_per_sec": tokens_per_sec,
        "mfu": mfu,
    }''',
    "demo": """P, L, d, heads, s = 6738415616, 32, 4096, 32, 2048
peak, bw = 312e12, 2.0e12  # A100-like: 312 TFLOP/s bf16, 2 TB/s HBM
print(f"ridge point = {peak / bw:.0f} FLOP/byte")
print(f"{'batch':>6} {'AI':>8} {'bound':>8} {'step ms':>8} {'tok/s':>9} {'mfu':>7}")
for batch in [1, 8, 32, 64, 128, 256]:
    out = inference_roofline(P, L, d, heads, heads, s, batch, peak, bw)
    print(f"{batch:>6} {out['arithmetic_intensity']:>8.1f} {out['bound']:>8} "
          f"{out['step_time'] * 1e3:>8.2f} {out['tokens_per_sec']:>9.1f} {out['mfu']:>7.3f}")

gqa = inference_roofline(P, L, d, heads, 8, s, 64, peak, bw)
mha = inference_roofline(P, L, d, heads, heads, s, 64, peak, bw)
print(f"batch 64 KV cache: MHA {mha['kv_cache_bytes'] / 2**30:.1f} GiB vs GQA-8 {gqa['kv_cache_bytes'] / 2**30:.1f} GiB")""",
}
