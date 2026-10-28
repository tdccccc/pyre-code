"""INT4 Group-wise Quantization task."""

TASK = {
    "title": "INT4 Group-wise Quantization",
    "title_zh": "INT4 分组量化",
    "difficulty": "Hard",
    "description_en": "Implement INT4 group-wise asymmetric weight quantization with nibble packing, the storage format behind GPTQ and AWQ.\n\nEach row of a weight matrix is split into contiguous groups of `group_size` columns. Every group gets its own scale and zero point, so a group with a large range does not reduce the precision of a group with a small range. Two 4-bit codes are packed into each byte.\n\n**Signature:** `int4_group_quantize(w, group_size) -> tuple[Tensor, Tensor, Tensor]`\n\n**Parameters:**\n- `w` — float32 tensor `(out_features, in_features)`; $\\texttt{in\\_features}$ divisible by $\\texttt{group\\_size}$; $\\texttt{group\\_size}$ even\n- `group_size` — number of consecutive columns per group\n\n**Returns:** `(packed, scales, zeros)`\n- `packed` — uint8 `(out_features, in_features // 2)`\n- `scales` — float32 `(out_features, in_features // group_size)`\n- `zeros` — uint8 `(out_features, in_features // group_size)`\n\n**Formula:** for every row and every group:\n$$\n\\begin{aligned}\n\\texttt{mn} &= \\min\\left(\\min(\\text{group}),\\ 0\\right) \\\\\n\\texttt{mx} &= \\max\\left(\\max(\\text{group}),\\ 0\\right) \\\\\n\\texttt{scale} &= \\begin{cases} (\\texttt{mx} - \\texttt{mn})/15 & \\text{if } \\texttt{mx} > \\texttt{mn} \\\\ 1.0 & \\text{if } \\texttt{mx} = \\texttt{mn} \\end{cases} \\\\\n\\texttt{zero} &= \\operatorname{clamp}\\left(\\operatorname{round}(-\\texttt{mn}/\\texttt{scale}),\\ 0,\\ 15\\right) \\\\\nq &= \\operatorname{clamp}\\left(\\operatorname{round}(w/\\texttt{scale}) + \\texttt{zero},\\ 0,\\ 15\\right)\n\\end{aligned}\n$$\n`round` is `torch.round` (half to even). Packing: along each row, element $2i$ goes into the low nibble and element $2i+1$ into the high nibble of byte $i$: `packed = q[:, 0::2] | (q[:, 1::2] << 4)`.\n\nDequantization: $\\hat{w} = (q - \\texttt{zero}) \\cdot \\texttt{scale}$, where $q$ is the unpacked 4-bit code and `zero`, `scale` are those of the element's own group.\n\n**Why 0 is in the range:** including 0 in $[\\texttt{mn}, \\texttt{mx}]$ makes the zero point an integer that is exactly representable, so the real value 0 dequantizes to exactly 0. It also keeps the zero point inside $[0, 15]$, so a one-signed group (all positive or all negative) is not saturated.\n\n**Constraints:**\n- Each group is quantized independently; `mn` and `mx` come from that group only (together with 0), never across the row\n- An all-zero group has $\\texttt{scale} = 0$ and uses $\\texttt{scale} = 1.0$\n- Do not modify the input tensor `w`",
    "description_zh": "实现 INT4 分组非对称权重量化及 nibble 打包，这是 GPTQ、AWQ 等方法使用的存储格式。\n\n权重矩阵的每一行被切分为连续的 `group_size` 列为一组。每组拥有独立的 scale 和 zero point，因此数值范围大的组不会降低数值范围小的组的精度。每个字节打包两个 4 位编码。\n\n**签名:** `int4_group_quantize(w, group_size) -> tuple[Tensor, Tensor, Tensor]`\n\n**参数:**\n- `w` — float32 张量 `(out_features, in_features)`；$\\texttt{in\\_features}$ 可被 $\\texttt{group\\_size}$ 整除；$\\texttt{group\\_size}$ 为偶数\n- `group_size` — 每组包含的连续列数\n\n**返回:** `(packed, scales, zeros)`\n- `packed` — uint8 `(out_features, in_features // 2)`\n- `scales` — float32 `(out_features, in_features // group_size)`\n- `zeros` — uint8 `(out_features, in_features // group_size)`\n\n**公式:** 对每一行的每一组：\n$$\n\\begin{aligned}\n\\texttt{mn} &= \\min\\left(\\min(\\text{该组}),\\ 0\\right) \\\\\n\\texttt{mx} &= \\max\\left(\\max(\\text{该组}),\\ 0\\right) \\\\\n\\texttt{scale} &= \\begin{cases} (\\texttt{mx} - \\texttt{mn})/15 & \\text{若 } \\texttt{mx} > \\texttt{mn} \\\\ 1.0 & \\text{若 } \\texttt{mx} = \\texttt{mn} \\end{cases} \\\\\n\\texttt{zero} &= \\operatorname{clamp}\\left(\\operatorname{round}(-\\texttt{mn}/\\texttt{scale}),\\ 0,\\ 15\\right) \\\\\nq &= \\operatorname{clamp}\\left(\\operatorname{round}(w/\\texttt{scale}) + \\texttt{zero},\\ 0,\\ 15\\right)\n\\end{aligned}\n$$\n`round` 为 `torch.round`（向偶数舍入）。打包：沿每一行，元素 $2i$ 放入字节 $i$ 的低 4 位，元素 $2i+1$ 放入高 4 位：`packed = q[:, 0::2] | (q[:, 1::2] << 4)`。\n\n反量化：$\\hat{w} = (q - \\texttt{zero}) \\cdot \\texttt{scale}$，其中 $q$ 为解包后的 4 位编码，`zero` 与 `scale` 为该元素所在组的参数。\n\n**为什么取值范围包含 0:** 将 0 纳入 $[\\texttt{mn}, \\texttt{mx}]$ 后，zero point 为可被精确表示的整数，实数 0 反量化后恰好为 0。同时 zero point 始终落在 $[0, 15]$ 内，因此全正或全负的组不会发生饱和截断。\n\n**约束:**\n- 每组独立量化；`mn` 与 `mx` 只取自该组（并包含 0），不跨行\n- 全零组的 $\\texttt{scale} = 0$，此时使用 $\\texttt{scale} = 1.0$\n- 不得修改输入张量 `w`",
    "function_name": "int4_group_quantize",
    "hint": "g = w.reshape(out, in // gs, gs)\nmn = g.amin(-1).clamp(max=0); mx = g.amax(-1).clamp(min=0)   # range includes 0\nscale = (mx - mn) / 15  (1.0 where scale == 0)\nzero = clamp(round(-mn / scale), 0, 15)\nq = clamp(round(g / scale) + zero, 0, 15)                  # uint8\npacked = q[:, 0::2] | (q[:, 1::2] << 4)",
    "hint_zh": "g = w.reshape(out, in // gs, gs)\nmn = g.amin(-1).clamp(max=0); mx = g.amax(-1).clamp(min=0)   # 范围包含 0\nscale = (mx - mn) / 15  （scale == 0 处取 1.0）\nzero = clamp(round(-mn / scale), 0, 15)\nq = clamp(round(g / scale) + zero, 0, 15)                  # uint8\npacked = q[:, 0::2] | (q[:, 1::2] << 4)",
    "tests": [
        {
            "name": "Output dtypes and shapes",
            "code": """
import torch
torch.manual_seed(0)
w = torch.randn(4, 32)
packed, scales, zeros = {fn}(w, 8)
assert packed.dtype == torch.uint8, f'packed dtype should be uint8, got {packed.dtype}'
assert scales.dtype == torch.float32, f'scales dtype should be float32, got {scales.dtype}'
assert zeros.dtype == torch.uint8, f'zeros dtype should be uint8, got {zeros.dtype}'
assert packed.shape == (4, 16), f'packed shape should be (4, 16), got {tuple(packed.shape)}'
assert scales.shape == (4, 4), f'scales shape should be (4, 4), got {tuple(scales.shape)}'
assert zeros.shape == (4, 4), f'zeros shape should be (4, 4), got {tuple(zeros.shape)}'
""",
        },
        {
            "name": "Exact packing on a hand-made example",
            "code": """
import torch
# group 0 = [-5, 0, 3, 10]: range [-5, 10] -> scale 1.0, zero 5, codes [0, 5, 8, 15]
# group 1 = [-10, 0, 10, 20]: range [-10, 20] -> scale 2.0, zero 5, codes [0, 5, 10, 15]
w = torch.tensor([[-5.0, 0.0, 3.0, 10.0, -10.0, 0.0, 10.0, 20.0]])
packed, scales, zeros = {fn}(w, 4)
assert torch.equal(scales, torch.tensor([[1.0, 2.0]])), f'scales should be [[1.0, 2.0]], got {scales.tolist()}'
assert torch.equal(zeros, torch.tensor([[5, 5]], dtype=torch.uint8)), f'zeros should be [[5, 5]], got {zeros.tolist()}'
# byte = low | (high << 4): (0, 5) -> 0x50, (8, 15) -> 0xF8, (0, 5) -> 0x50, (10, 15) -> 0xFA
expected = torch.tensor([[0x50, 0xF8, 0x50, 0xFA]], dtype=torch.uint8)
assert torch.equal(packed, expected), f'packed should be {expected.tolist()}, got {packed.tolist()}'
""",
        },
        {
            "name": "Unpacked nibbles are in [0, 15] and match expected codes",
            "code": """
import torch
w = torch.tensor([[-5.0, 0.0, 3.0, 10.0, -10.0, 0.0, 10.0, 20.0]])
packed, _, _ = {fn}(w, 4)
low = packed & 0x0F    # even elements
high = packed >> 4     # odd elements
q = torch.stack([low, high], dim=-1).reshape(1, 8).long()
assert q.min().item() >= 0 and q.max().item() <= 15, f'codes out of [0, 15]: {q.tolist()}'
expected = [[0, 5, 8, 15, 0, 5, 10, 15]]
assert q.tolist() == expected, f'codes should be {expected}, got {q.tolist()}'
""",
        },
        {
            "name": "Random reconstruction error within scale/2",
            "code": """
import torch
torch.manual_seed(1)
out_f, in_f, g = 16, 64, 16
w = torch.randn(out_f, in_f)
packed, scales, zeros = {fn}(w, g)
low = packed & 0x0F
high = packed >> 4
q = torch.stack([low, high], dim=-1).reshape(out_f, in_f).float()
assert q.min().item() >= 0 and q.max().item() <= 15, 'codes must be in [0, 15]'
z = zeros.float().repeat_interleave(g, dim=1)
s = scales.repeat_interleave(g, dim=1)
w_hat = (q - z) * s
err = (w_hat - w).abs()
bound = s / 2 + 1e-6
assert bool((err <= bound).all()), f'max error {err.max().item():.4f} exceeds scale/2 bound'
""",
        },
        {
            "name": "Matches explicit per-group reference",
            "code": """
import torch
torch.manual_seed(2)
out_f, in_f, g = 8, 32, 8
w = torch.randn(out_f, in_f) * 3
packed, scales, zeros = {fn}(w, g)
ref_q = torch.zeros(out_f, in_f, dtype=torch.long)
ref_s = torch.zeros(out_f, in_f // g)
ref_z = torch.zeros(out_f, in_f // g, dtype=torch.long)
for r in range(out_f):
    for k in range(in_f // g):
        seg = w[r, k * g:(k + 1) * g]
        mn = seg.min().clamp(max=0)
        mx = seg.max().clamp(min=0)
        sc = (mx - mn) / 15
        if sc == 0:
            sc = torch.tensor(1.0)
        zp = int(torch.clamp(torch.round(-mn / sc), 0, 15))
        qq = torch.clamp(torch.round(seg / sc) + zp, 0, 15)
        ref_q[r, k * g:(k + 1) * g] = qq.long()
        ref_s[r, k] = sc
        ref_z[r, k] = zp
low = packed & 0x0F
high = packed >> 4
q = torch.stack([low, high], dim=-1).reshape(out_f, in_f).long()
assert torch.allclose(scales, ref_s, rtol=1e-6), 'scales differ from reference'
assert torch.equal(zeros.long(), ref_z), f'zeros differ from reference'
assert torch.equal(q, ref_q), 'codes differ from reference'
""",
        },
        {
            "name": "Group boundaries: each group uses its own range",
            "code": """
import torch
torch.manual_seed(3)
g = 8
small = torch.randn(1, g) * 0.01
large = torch.randn(1, g) * 100.0
w = torch.cat([small, large], dim=1)   # (1, 16): two groups with very different ranges
packed, scales, zeros = {fn}(w, g)
exp_small = (small.max().clamp(min=0) - small.min().clamp(max=0)) / 15
exp_large = (large.max().clamp(min=0) - large.min().clamp(max=0)) / 15
assert torch.allclose(scales[0, 0], exp_small, rtol=1e-5), f'group 0 scale {scales[0, 0].item()} should be {exp_small.item()}'
assert torch.allclose(scales[0, 1], exp_large, rtol=1e-5), f'group 1 scale {scales[0, 1].item()} should be {exp_large.item()}'
low = packed & 0x0F
high = packed >> 4
q = torch.stack([low, high], dim=-1).reshape(1, 2 * g).float()
s = scales.repeat_interleave(g, dim=1)
z = zeros.float().repeat_interleave(g, dim=1)
err = ((q - z) * s - w).abs()
assert bool((err <= s / 2 + 1e-6).all()), 'per-element error exceeds scale/2 of its own group'
""",
        },
        {
            "name": "All-zero group uses scale 1.0",
            "code": """
import torch
w = torch.tensor([[0.0, 0.0, 0.0, 0.0, 15.0, 15.0, 15.0, 15.0]])
packed, scales, zeros = {fn}(w, 4)
# group 0: range [0, 0] -> scale 0 -> 1.0, zero 0, codes 0
# group 1: range [0, 15] -> scale 1.0 (from the formula), zero 0, codes 15
assert torch.equal(scales, torch.ones(1, 2)), f'scales should be [[1.0, 1.0]], got {scales.tolist()}'
assert zeros.tolist() == [[0, 0]], f'zeros should be [[0, 0]], got {zeros.tolist()}'
assert packed.tolist() == [[0x00, 0x00, 0xFF, 0xFF]], f'packed should be [[0, 0, 255, 255]], got {packed.tolist()}'
""",
        },
        {
            "name": "One-signed groups reconstruct within scale/2",
            "code": """
import torch
# group 0 is all non-negative: range [0, 3] -> scale 0.2, zero 0
# group 1 is all non-positive: range [-4, 0] -> scale 4/15, zero 15
w = torch.tensor([[0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -1.5, 0.0]])
packed, scales, zeros = {fn}(w, 4)
assert torch.allclose(scales, torch.tensor([[0.2, 4 / 15]]), rtol=1e-6), f'scales should be [[0.2, 4/15]], got {scales.tolist()}'
assert zeros.tolist() == [[0, 15]], f'zeros should be [[0, 15]], got {zeros.tolist()}'
assert packed.tolist() == [[0x50, 0xFA, 0x40, 0xF9]], f'packed should be [[80, 250, 64, 249]], got {packed.tolist()}'
low = packed & 0x0F
high = packed >> 4
q = torch.stack([low, high], dim=-1).reshape(1, 8).float()
s = scales.repeat_interleave(4, dim=1)
z = zeros.float().repeat_interleave(4, dim=1)
w_hat = (q - z) * s
assert w_hat[0, 0].item() == 0.0 and w_hat[0, 7].item() == 0.0, 'the real value 0 must dequantize to exactly 0'
assert bool(((w_hat - w).abs() <= s / 2 + 1e-6).all()), 'one-signed group error exceeds scale/2'
# random one-signed groups: all-positive then all-negative, per row
torch.manual_seed(5)
pos = torch.rand(4, 16) * 5 + 1
neg = -(torch.rand(4, 16) * 5 + 1)
w2 = torch.cat([pos, neg], dim=1)
packed2, scales2, zeros2 = {fn}(w2, 16)
assert (zeros2[:, 0] == 0).all(), f'all-positive groups need zero 0, got {zeros2.tolist()}'
assert (zeros2[:, 1] == 15).all(), f'all-negative groups need zero 15, got {zeros2.tolist()}'
low2 = packed2 & 0x0F
high2 = packed2 >> 4
q2 = torch.stack([low2, high2], dim=-1).reshape(4, 32).float()
s2 = scales2.repeat_interleave(16, dim=1)
z2 = zeros2.float().repeat_interleave(16, dim=1)
err2 = ((q2 - z2) * s2 - w2).abs()
assert bool((err2 <= s2 / 2 + 1e-6).all()), f'random one-signed error {err2.max().item():.4f} exceeds scale/2'
""",
        },
        {
            "name": "Input is not mutated",
            "code": """
import torch
torch.manual_seed(4)
w = torch.randn(4, 16)
orig = w.clone()
{fn}(w, 8)
assert torch.equal(w, orig), 'input tensor was modified'
""",
        },
    ],
    "solution": '''def int4_group_quantize(w, group_size):
    out_features, in_features = w.shape
    n_groups = in_features // group_size

    # View each row as contiguous groups: (out, n_groups, group_size)
    wg = w.reshape(out_features, n_groups, group_size)
    # The range always includes 0, so the zero point is an exact integer in [0, 15]
    mn = wg.amin(dim=-1, keepdim=True).clamp(max=0)
    mx = wg.amax(dim=-1, keepdim=True).clamp(min=0)

    scale = (mx - mn) / 15.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # all-zero group
    zero = torch.clamp(torch.round(-mn / scale), 0, 15)

    q = torch.clamp(torch.round(wg / scale) + zero, 0, 15)
    q = q.reshape(out_features, in_features).to(torch.uint8)

    # Element 2i goes to the low nibble of byte i, element 2i+1 to the high nibble
    packed = q[:, 0::2] | (q[:, 1::2] << 4)

    scales = scale.squeeze(-1)
    zeros = zero.squeeze(-1).to(torch.uint8)
    return packed, scales, zeros''',
    "demo": """torch.manual_seed(0)
out_f, in_f, g = 4, 16, 8
w = torch.randn(out_f, in_f)

packed, scales, zeros = int4_group_quantize(w, g)
print('packed:', tuple(packed.shape), packed.dtype)
print('scales:', tuple(scales.shape), scales.dtype)
print('zeros: ', tuple(zeros.shape), zeros.dtype)

# Unpack and dequantize
low = packed & 0x0F
high = packed >> 4
q = torch.stack([low, high], dim=-1).reshape(out_f, in_f).float()
w_hat = (q - zeros.float().repeat_interleave(g, dim=1)) * scales.repeat_interleave(g, dim=1)
print('max abs error:', (w_hat - w).abs().max().item())
print('compression vs fp32:', w.numel() * 4 / (packed.numel() + scales.numel() * 4 + zeros.numel()))""",

}
