"""Training Memory Estimate (ZeRO) task."""

TASK = {
    "title": "Training Memory Estimate (ZeRO)",
    "title_zh": "训练显存估算（ZeRO）",
    "difficulty": "Medium",
    "description_en": "Estimate the per-GPU memory needed to train a model with mixed precision (bf16) and Adam. Model states are split across data-parallel ranks by ZeRO, while activations are not, and they often decide whether a job fits at all. Knowing which term dominates tells you whether to shard, to recompute, or to add micro-batch parallelism.\n\n**Signature:** `training_memory(num_params, n_layers, d_model, n_heads, seq_len, micro_batch, zero_stage=0, world_size=1, recompute=False) -> dict`\n\n**Parameters:**\n- `num_params` — $\\Psi$, total trainable parameters\n- `n_layers`, `d_model`, `n_heads` — architecture; $a = \\texttt{n\\_heads}$, $h = \\texttt{d\\_model}$\n- `seq_len` — $s$, tokens per sequence\n- `micro_batch` — $b$, sequences per GPU per micro-step\n- `zero_stage` — 0, 1, 2 or 3 (ZeRO stage); any other value raises `ValueError`\n- `world_size` — $N$, number of data-parallel ranks\n- `recompute` — if True, use full activation checkpointing\n\n**Model states (bytes, bf16 mixed precision with Adam):**\n- bf16 parameters: $2\\Psi$\n- bf16 gradients: $2\\Psi$\n- optimizer states: $12\\Psi$ (fp32 master weights 4 bytes + Adam m 4 bytes + Adam v 4 bytes)\n\n**ZeRO partitioning (each term is divided by $N$ only if the stage listed):**\n- stage 0: nothing is partitioned\n- stage 1: optimizer states\n- stage 2: optimizer states and gradients\n- stage 3: parameters, gradients and optimizer states\n\n**Activations (bytes, per GPU, no tensor or sequence parallelism, bf16):**\n- without recompute, per layer: $s \\cdot b \\cdot h \\cdot \\left(34 + \\frac{5 \\times a \\cdot s}{h}\\right)$ (Korthikanti et al. 2022)\n- with `recompute=True`, per layer: $2 \\times s \\cdot b \\cdot h$ (only each layer's input is kept)\n- multiply by `n_layers`. Activations are **not** divided by `world_size`.\n\n**Returns:** a dict of Python `float`s (bytes): `params`, `grads`, `optimizer`, `activations`, `total`. `total` is the sum of the other four.\n\n**Worked example:** `num_params=1000, n_layers=1, d_model=4, n_heads=2, seq_len=2, micro_batch=1`, stage 0, `world_size=1`, no recompute:\n- $\\texttt{params} = 2000$, $\\texttt{grads} = 2000$, $\\texttt{optimizer} = 12000$\n- $\\texttt{activations} = 2 \\times 1 \\times 4 \\times \\left(34 + \\frac{5 \\times 2 \\times 2}{4}\\right) = 8 \\times 39 = 312$\n- $\\texttt{total} = 16312$\n\n**Constraints:** ZeRO's model-state savings only apply to model states; activation memory is the same on every rank for a given micro-batch.",
    "description_zh": "估算使用混合精度（bf16）和 Adam 训练模型时，每张 GPU 所需的显存。模型状态由 ZeRO 在数据并行 rank 之间切分，而激活值不会被切分，它们往往决定任务能否放得下。看清哪一项占主导，才能决定是切分、重计算，还是减小 micro-batch。\n\n**签名:** `training_memory(num_params, n_layers, d_model, n_heads, seq_len, micro_batch, zero_stage=0, world_size=1, recompute=False) -> dict`\n\n**参数:**\n- `num_params` — $\\Psi$，可训练参数总数\n- `n_layers`、`d_model`、`n_heads` — 模型结构；$a = \\texttt{n\\_heads}$，$h = \\texttt{d\\_model}$\n- `seq_len` — $s$，每个序列的 token 数\n- `micro_batch` — $b$，每张 GPU 每个 micro-step 的序列数\n- `zero_stage` — ZeRO 阶段，取值 0、1、2 或 3；其他取值抛出 `ValueError`\n- `world_size` — $N$，数据并行 rank 数\n- `recompute` — 为 True 时使用完整的激活重计算（activation checkpointing）\n\n**模型状态（字节，bf16 混合精度 + Adam）:**\n- bf16 参数：$2\\Psi$\n- bf16 梯度：$2\\Psi$\n- 优化器状态：$12\\Psi$（fp32 主权重 4 字节 + Adam 的 m 4 字节 + Adam 的 v 4 字节）\n\n**ZeRO 切分（仅当列出的阶段时，该项才除以 $N$）:**\n- 阶段 0：不切分任何内容\n- 阶段 1：优化器状态\n- 阶段 2：优化器状态和梯度\n- 阶段 3：参数、梯度和优化器状态\n\n**激活值（字节，每张 GPU，无张量并行或序列并行，bf16）:**\n- 不重计算时，每层：$s \\cdot b \\cdot h \\cdot \\left(34 + \\frac{5 \\times a \\cdot s}{h}\\right)$（Korthikanti et al. 2022）\n- `recompute=True` 时，每层：$2 \\times s \\cdot b \\cdot h$（只保留每层的输入）\n- 再乘以 `n_layers`。激活值**不**除以 `world_size`。\n\n**返回:** 由 Python `float`（字节数）组成的字典：`params`、`grads`、`optimizer`、`activations`、`total`。`total` 为其余四项之和。\n\n**手算示例:** `num_params=1000, n_layers=1, d_model=4, n_heads=2, seq_len=2, micro_batch=1`，阶段 0，`world_size=1`，不重计算：\n- $\\texttt{params} = 2000$，$\\texttt{grads} = 2000$，$\\texttt{optimizer} = 12000$\n- $\\texttt{activations} = 2 \\times 1 \\times 4 \\times \\left(34 + \\frac{5 \\times 2 \\times 2}{4}\\right) = 8 \\times 39 = 312$\n- $\\texttt{total} = 16312$\n\n**约束:** ZeRO 的节省只作用于模型状态；对于给定的 micro-batch，激活值在每个 rank 上是相同的。",
    "function_name": "training_memory",
    "hint": "if zero_stage not in (0,1,2,3): raise ValueError\nparam_div = N if stage==3 else 1\ngrad_div  = N if stage>=2 else 1\nopt_div   = N if stage>=1 else 1\nparams = 2*Psi/param_div;  grads = 2*Psi/grad_div;  optimizer = 12*Psi/opt_div\nper_layer = 2*s*b*h if recompute else s*b*h*(34 + 5*a*s/h)\nactivations = n_layers * per_layer   # not divided by N\ntotal = params + grads + optimizer + activations",
    "hint_zh": "if zero_stage not in (0,1,2,3): 抛出 ValueError\nparam_div = N if stage==3 else 1\ngrad_div  = N if stage>=2 else 1\nopt_div   = N if stage>=1 else 1\nparams = 2*Psi/param_div;  grads = 2*Psi/grad_div;  optimizer = 12*Psi/opt_div\nper_layer = 2*s*b*h if recompute else s*b*h*(34 + 5*a*s/h)\nactivations = n_layers * per_layer   # 不除以 N\ntotal = params + grads + optimizer + activations",
    "tests": [
        {
            "name": "Returns floats with all keys",
            "code": """
import torch
out = {fn}(1000, 1, 4, 2, 2, 1)
for k in ("params", "grads", "optimizer", "activations", "total"):
    assert k in out, f'Missing key {k}'
    assert type(out[k]) is float, f'{k} should be float, got {type(out[k]).__name__}'
""",
        },
        {
            "name": "Tiny hand-computed example",
            "code": """
import torch
# Psi=1000: params 2000, grads 2000, optimizer 12000
# activations: s*b*h*(34 + 5*a*s/h) = 2*1*4*(34 + 5*2*2/4) = 312
out = {fn}(1000, 1, 4, 2, 2, 1)
assert out["params"] == 2000.0, f'params: expected 2000.0, got {out["params"]}'
assert out["grads"] == 2000.0, f'grads: expected 2000.0, got {out["grads"]}'
assert out["optimizer"] == 12000.0, f'optimizer: expected 12000.0, got {out["optimizer"]}'
assert out["activations"] == 312.0, f'activations: expected 312.0, got {out["activations"]}'
assert out["total"] == 16312.0, f'total: expected 16312.0, got {out["total"]}'
""",
        },
        {
            "name": "7B model, ZeRO stage 0 (N=8)",
            "code": """
import torch
import math
# Psi = 7e9, N = 8, L=32, h=4096, a=32, s=2048, b=1
# model states: params 14e9, grads 14e9, optimizer 84e9 (16 bytes/param, nothing divided)
# activations: 32 * 2048*4096 * (34 + 5*32*2048/4096) = 32 * 8388608 * 114 = 30601641984
out = {fn}(7_000_000_000, 32, 4096, 32, 2048, 1, zero_stage=0, world_size=8)
expected = {"params": 14e9, "grads": 14e9, "optimizer": 84e9, "activations": 30601641984.0, "total": 142601641984.0}
for k, v in expected.items():
    assert math.isclose(out[k], v, rel_tol=1e-9), f'stage 0 {k}: expected {v}, got {out[k]}'
""",
        },
        {
            "name": "7B model, ZeRO stages 1-3 (N=8)",
            "code": """
import torch
import math
# Same model as the stage-0 test; only the model-state terms change.
# stage 1: optimizer 84e9/8 = 10.5e9                       -> total 69101641984
# stage 2: optimizer 10.5e9, grads 14e9/8 = 1.75e9          -> total 56851641984
# stage 3: params, grads, optimizer all /8                  -> total 44601641984
cases = {
    1: {"params": 14e9, "grads": 14e9, "optimizer": 10.5e9, "total": 69101641984.0},
    2: {"params": 14e9, "grads": 1.75e9, "optimizer": 10.5e9, "total": 56851641984.0},
    3: {"params": 1.75e9, "grads": 1.75e9, "optimizer": 10.5e9, "total": 44601641984.0},
}
for stage, expected in cases.items():
    out = {fn}(7_000_000_000, 32, 4096, 32, 2048, 1, zero_stage=stage, world_size=8)
    assert out["activations"] == 30601641984.0, f'stage {stage}: activations should be unchanged, got {out["activations"]}'
    for k, v in expected.items():
        assert math.isclose(out[k], v, rel_tol=1e-9), f'stage {stage} {k}: expected {v}, got {out[k]}'
""",
        },
        {
            "name": "Stage 0 model states are 16*Psi; stage 3 divides them by N",
            "code": """
import torch
import math
torch.manual_seed(0)
for _ in range(4):
    psi = int(torch.randint(10**8, 10**10, (1,)))
    N = int(torch.randint(1, 65, (1,)))
    out0 = {fn}(psi, 2, 256, 4, 128, 1, zero_stage=0, world_size=N)
    states0 = out0["params"] + out0["grads"] + out0["optimizer"]
    assert math.isclose(states0, 16 * psi, rel_tol=1e-9), f'stage 0 model states should be 16*Psi={16 * psi}, got {states0}'
    out3 = {fn}(psi, 2, 256, 4, 128, 1, zero_stage=3, world_size=N)
    states3 = out3["params"] + out3["grads"] + out3["optimizer"]
    assert math.isclose(states3, 16 * psi / N, rel_tol=1e-9), f'stage 3 (N={N}) model states should be 16*Psi/N={16 * psi / N}, got {states3}'
""",
        },
        {
            "name": "Recompute keeps only layer inputs",
            "code": """
import torch
import math
# Per layer 2*s*b*h = 2*2048*4096 = 16777216 bytes; x32 layers = 536870912
full = {fn}(7_000_000_000, 32, 4096, 32, 2048, 1)
rec = {fn}(7_000_000_000, 32, 4096, 32, 2048, 1, recompute=True)
assert rec["activations"] == 536870912.0, f'recompute activations: expected 536870912.0, got {rec["activations"]}'
assert math.isclose(full["activations"] / rec["activations"], 57.0, rel_tol=1e-9), f'full/recompute ratio should be 57, got {full["activations"] / rec["activations"]}'
assert rec["total"] == 112e9 + 536870912.0, f'recompute total: expected {112e9 + 536870912.0}, got {rec["total"]}'
assert rec["params"] == full["params"] and rec["grads"] == full["grads"] and rec["optimizer"] == full["optimizer"], 'recompute must not change model states'
""",
        },
        {
            "name": "Activations ignore zero_stage and world_size",
            "code": """
import torch
ref = {fn}(3_000_000_000, 24, 2048, 16, 1024, 2)["activations"]
assert ref == 24 * 2 * 1024 * 2048 * (34 + 5 * 16 * 1024 / 2048), f'reference activations wrong: {ref}'
for stage, N in [(0, 1), (1, 8), (2, 8), (3, 64)]:
    out = {fn}(3_000_000_000, 24, 2048, 16, 1024, 2, zero_stage=stage, world_size=N)
    assert out["activations"] == ref, f'stage {stage}, N={N}: activations {out["activations"]} should equal {ref} (not divided by N)'
""",
        },
        {
            "name": "Invalid zero_stage raises ValueError",
            "code": """
import torch
for bad in (4, -1):
    raised = False
    try:
        {fn}(1000, 1, 4, 2, 2, 1, zero_stage=bad)
    except ValueError:
        raised = True
    assert raised, f'zero_stage={bad} should raise ValueError'
""",
        },
        {
            "name": "Random configs match reference formula",
            "code": """
import torch
import math
torch.manual_seed(3)
for _ in range(5):
    psi = int(torch.randint(10**8, 10**11, (1,)))
    L = int(torch.randint(1, 81, (1,)))
    a = int(torch.randint(4, 65, (1,)))
    h = a * 64 * int(torch.randint(1, 3, (1,)))
    s = int(torch.randint(128, 8193, (1,)))
    b = int(torch.randint(1, 9, (1,)))
    stage = int(torch.randint(0, 4, (1,)))
    N = int(torch.randint(1, 129, (1,)))
    rec = bool(torch.randint(0, 2, (1,)))
    out = {fn}(psi, L, h, a, s, b, zero_stage=stage, world_size=N, recompute=rec)
    div_p = N if stage == 3 else 1
    div_g = N if stage in (2, 3) else 1
    div_o = N if stage >= 1 else 1
    p_ref = 2 * psi / div_p
    g_ref = 2 * psi / div_g
    o_ref = 12 * psi / div_o
    per_layer = 2 * s * b * h if rec else s * b * h * (34 + 5 * a * s / h)
    a_ref = L * per_layer
    cfg = f'Psi={psi} L={L} h={h} a={a} s={s} b={b} stage={stage} N={N} recompute={rec}'
    assert math.isclose(out["params"], p_ref, rel_tol=1e-9), f'{cfg}: params {out["params"]} != {p_ref}'
    assert math.isclose(out["grads"], g_ref, rel_tol=1e-9), f'{cfg}: grads {out["grads"]} != {g_ref}'
    assert math.isclose(out["optimizer"], o_ref, rel_tol=1e-9), f'{cfg}: optimizer {out["optimizer"]} != {o_ref}'
    assert math.isclose(out["activations"], a_ref, rel_tol=1e-9), f'{cfg}: activations {out["activations"]} != {a_ref}'
    assert math.isclose(out["total"], p_ref + g_ref + o_ref + a_ref, rel_tol=1e-9), f'{cfg}: total {out["total"]} != {p_ref + g_ref + o_ref + a_ref}'
""",
        },
    ],
    "solution": '''def training_memory(num_params, n_layers, d_model, n_heads, seq_len, micro_batch, zero_stage=0, world_size=1, recompute=False):
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2 or 3, got {zero_stage}")

    # Model states: bf16 params (2) + bf16 grads (2) + fp32 master/m/v (12) bytes per parameter
    param_div = world_size if zero_stage == 3 else 1
    grad_div = world_size if zero_stage >= 2 else 1
    opt_div = world_size if zero_stage >= 1 else 1
    params = 2 * num_params / param_div
    grads = 2 * num_params / grad_div
    optimizer = 12 * num_params / opt_div

    # Activations per layer (Korthikanti et al. 2022, bf16, no tensor/sequence parallelism)
    s, b, h, a = seq_len, micro_batch, d_model, n_heads
    if recompute:
        # Full checkpointing: only each layer's input tensor is kept
        per_layer = 2 * s * b * h
    else:
        per_layer = s * b * h * (34 + 5 * a * s / h)
    activations = float(n_layers * per_layer)  # not sharded across ranks

    total = params + grads + optimizer + activations
    return {
        "params": float(params),
        "grads": float(grads),
        "optimizer": float(optimizer),
        "activations": activations,
        "total": float(total),
    }''',
    "demo": """Psi, L, h, a, s, b, N = 7_000_000_000, 32, 4096, 32, 2048, 1, 8
GiB = 2**30
print(f"7B model, s={s}, micro-batch={b}, N={N} GPUs (GiB per GPU)")
for stage in range(4):
    m = training_memory(Psi, L, h, a, s, b, zero_stage=stage, world_size=N)
    print(f"  ZeRO-{stage}: model states {(m['params'] + m['grads'] + m['optimizer']) / GiB:6.1f}  "
          f"activations {m['activations'] / GiB:5.1f}  total {m['total'] / GiB:6.1f}")

rec = training_memory(Psi, L, h, a, s, b, zero_stage=3, world_size=N, recompute=True)
print(f"ZeRO-3 with recompute: activations {rec['activations'] / GiB:.2f} GiB, total {rec['total'] / GiB:.1f} GiB")""",
}
