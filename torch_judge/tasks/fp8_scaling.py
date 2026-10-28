"""FP8 Delayed Scaling task."""

TASK = {
    "title": "FP8 Delayed Scaling",
    "title_zh": "FP8 延迟缩放",
    "difficulty": "Medium",
    "description_en": "Implement Transformer-Engine-style delayed scaling for FP8 (E4M3) tensors.\n\nComputing a scale from the tensor you are about to quantize needs a full pass over it first. Delayed scaling avoids that: the scale used now is derived from the amax (max absolute value) history of previous steps, while the current tensor's amax is recorded for future steps.\n\n**Signature:** `DelayedScaler(history_len=16, fp8_max=448.0, margin=0)` (class) with methods `quantize(x) -> tuple[Tensor, float]` and `dequantize(x_fp8, scale) -> Tensor`\n\n**Parameters:**\n- `history_len` — number of most recent amax values kept in the history\n- `fp8_max` — largest magnitude representable in E4M3 (448.0)\n- `margin` — headroom in powers of two; the scale is divided by $2^{\\texttt{margin}}$\n\n**State:** `scale` (Python float, starts at 1.0) and an amax history of the most recent `history_len` values (empty at start).\n\n**Formula (quantize):**\n```\ns = self.scale                                   # scale BEFORE this call's update\nx_fp8 = (x * s).clamp(-fp8_max, fp8_max).to(float8_e4m3fn)\nhistory = (history + [x.abs().max()])[-history_len:]\nif max(history) > 0:\n    self.scale = fp8_max / max(history) / 2 ** margin\nreturn x_fp8, s\n```\nDequantization: `x_fp8.to(torch.float32) / scale`.\n\n**Returns:**\n- `quantize` returns `(x_fp8, s)`: `x_fp8` is a `torch.float8_e4m3fn` tensor with the shape of `x`, `s` is the Python float scale used for this call\n- `dequantize` returns a float32 tensor\n\n**Constraints:**\n- The first call uses scale 1.0\n- amax is the maximum of absolute values, and the current tensor's amax is included in the history after it is used\n- If every amax in the history is 0, the scale stays unchanged\n- Do not modify the input tensor `x`",
    "description_zh": "实现 Transformer Engine 风格的 FP8（E4M3）延迟缩放。\n\n若要对张量做量化，直接根据该张量计算 scale 需要先完整遍历一遍它。延迟缩放避免了这一步：当前使用的 scale 由之前若干步的 amax（绝对值最大值）历史推导，而当前张量的 amax 则被记录下来，供后续步使用。\n\n**签名:** `DelayedScaler(history_len=16, fp8_max=448.0, margin=0)`（类），方法为 `quantize(x) -> tuple[Tensor, float]` 与 `dequantize(x_fp8, scale) -> Tensor`\n\n**参数:**\n- `history_len` — 历史中保留的最近 amax 值个数\n- `fp8_max` — E4M3 可表示的最大幅值（448.0）\n- `margin` — 以 2 的幂为单位的余量；scale 除以 $2^{\\texttt{margin}}$\n\n**状态:** `scale`（Python float，初始为 1.0），以及保存最近 `history_len` 个值的 amax 历史（初始为空）。\n\n**公式（quantize）:**\n```\ns = self.scale                                   # 本次调用更新之前的 scale\nx_fp8 = (x * s).clamp(-fp8_max, fp8_max).to(float8_e4m3fn)\nhistory = (history + [x.abs().max()])[-history_len:]\nif max(history) > 0:\n    self.scale = fp8_max / max(history) / 2 ** margin\nreturn x_fp8, s\n```\n反量化：`x_fp8.to(torch.float32) / scale`。\n\n**返回:**\n- `quantize` 返回 `(x_fp8, s)`：`x_fp8` 为形状与 `x` 相同的 `torch.float8_e4m3fn` 张量，`s` 为本次调用所用的 Python float scale\n- `dequantize` 返回 float32 张量\n\n**约束:**\n- 第一次调用使用 scale 1.0\n- amax 取绝对值的最大值；当前张量的 amax 在被使用之后才计入历史\n- 若历史中所有 amax 均为 0，则 scale 保持不变\n- 不得修改输入张量 `x`",
    "function_name": "DelayedScaler",
    "hint": "class DelayedScaler:\n  quantize(x):\n    s = self.scale                              # from history, BEFORE this tensor\n    x_fp8 = (x * s).clamp(-448, 448).to(float8_e4m3fn)\n    history = (history + [x.abs().max()])[-history_len:]\n    if max(history) > 0: scale = 448 / max(history) / 2**margin\n    return x_fp8, s\n  dequantize(x_fp8, s): x_fp8.float() / s",
    "hint_zh": "class DelayedScaler:\n  quantize(x):\n    s = self.scale                              # 来自历史，本张量更新之前\n    x_fp8 = (x * s).clamp(-448, 448).to(float8_e4m3fn)\n    history = (history + [x.abs().max()])[-history_len:]\n    if max(history) > 0: scale = 448 / max(history) / 2**margin\n    return x_fp8, s\n  dequantize(x_fp8, s): x_fp8.float() / s",
    "tests": [
        {
            "name": "First call uses scale 1.0 and clamps to FP8 max",
            "code": """
import torch
sc = {fn}()
x = torch.tensor([1.0, -500.0, 256.0, 1000.0])
x_fp8, s = sc.quantize(x)
assert isinstance(s, float), f'scale should be a Python float, got {type(s)}'
assert s == 1.0, f'first call should use scale 1.0, got {s}'
assert x_fp8.dtype == torch.float8_e4m3fn, f'expected float8_e4m3fn, got {x_fp8.dtype}'
got = x_fp8.to(torch.float32).tolist()
assert got == [1.0, -448.0, 256.0, 448.0], f'values beyond 448 must clamp, got {got}'
""",
        },
        {
            "name": "Second call uses scale derived from first amax",
            "code": """
import torch
sc = {fn}()
sc.quantize(torch.tensor([1.0, -500.0, 256.0, 1000.0]))   # amax = 1000
x = torch.tensor([2.0, -4.0])
x_fp8, s = sc.quantize(x)
expected = 448.0 / 1000.0
assert abs(s - expected) < 1e-9, f'scale should be 448/1000 = {expected}, got {s}'
back = x_fp8.to(torch.float32) / expected
assert torch.allclose(back, x, rtol=0.07), f'values should round-trip at the new scale, got {back.tolist()}'
""",
        },
        {
            "name": "Absolute amax drives the scale",
            "code": """
import torch
sc = {fn}()
sc.quantize(torch.tensor([-3.0, 1.0]))   # amax = |-3| = 3
_, s = sc.quantize(torch.tensor([0.5]))
expected = 448.0 / 3.0
assert abs(s - expected) < 1e-9, f'amax should be |-3| = 3, so scale should be {expected}, got {s}'
""",
        },
        {
            "name": "History window drops old amax values",
            "code": """
import torch
sc = {fn}(history_len=2)
ones = torch.ones(4)
_, s1 = sc.quantize(torch.tensor([1000.0]))   # history [1000]
_, s2 = sc.quantize(ones)                     # uses 448/1000; history [1000, 1]
_, s3 = sc.quantize(ones)                     # spike still in window; history [1, 1] afterwards
_, s4 = sc.quantize(ones)                     # spike has left the window: uses 448/1
assert s1 == 1.0, f'first call should use 1.0, got {s1}'
assert abs(s2 - 0.448) < 1e-9, f'second call should use 448/1000, got {s2}'
assert abs(s3 - 0.448) < 1e-9, f'third call should still use 448/1000, got {s3}'
assert abs(s4 - 448.0) < 1e-9, f'after the spike leaves the window the scale should be 448, got {s4}'
""",
        },
        {
            "name": "Margin halves the scale per unit",
            "code": """
import torch
for m in range(3):
    sc = {fn}(margin=m)
    sc.quantize(torch.tensor([1000.0]))
    _, s = sc.quantize(torch.tensor([1.0]))
    expected = 448.0 / 1000.0 / (2 ** m)
    assert abs(s - expected) < 1e-9, f'margin={m}: expected scale {expected}, got {s}'
""",
        },
        {
            "name": "All-zero history keeps the scale",
            "code": """
import torch
sc = {fn}()
x_fp8, s1 = sc.quantize(torch.zeros(8))
assert s1 == 1.0, f'first call should use 1.0, got {s1}'
assert sc.scale == 1.0, f'scale must stay 1.0 when every amax is 0, got {sc.scale}'
_, s2 = sc.quantize(torch.zeros(3))
assert s2 == 1.0, f'scale must stay 1.0 after more zero inputs, got {s2}'
assert torch.all(x_fp8.to(torch.float32) == 0), 'zeros must quantize to zeros'
""",
        },
        {
            "name": "Dequantize divides by the given scale",
            "code": """
import torch
sc = {fn}()
q = torch.tensor([1.0, 2.0, -448.0, 0.5]).to(torch.float8_e4m3fn)
out = sc.dequantize(q, 2.0)
assert out.dtype == torch.float32, f'dequantize should return float32, got {out.dtype}'
assert out.tolist() == [0.5, 1.0, -224.0, 0.25], f'expected x_fp8 / scale, got {out.tolist()}'
""",
        },
        {
            "name": "Calibrated round trip has small relative error",
            "code": """
import torch
torch.manual_seed(0)
sc = {fn}()
x = torch.randn(1024)
sc.quantize(x)                 # calibrate the history on the same statistics
x_fp8, s = sc.quantize(x)
x_hat = sc.dequantize(x_fp8, s)
rel = ((x_hat - x).norm() / x.norm()).item()
assert rel < 0.07, f'relative round-trip error {rel:.4f} too large'
""",
        },
        {
            "name": "Input is not mutated",
            "code": """
import torch
torch.manual_seed(1)
sc = {fn}()
x = torch.randn(16) * 50
orig = x.clone()
sc.quantize(x)
sc.quantize(x)
assert torch.equal(x, orig), 'quantize must not modify its input'
""",
        },
    ],
    "solution": '''class DelayedScaler:
    def __init__(self, history_len=16, fp8_max=448.0, margin=0):
        self.history_len = history_len
        self.fp8_max = fp8_max
        self.margin = margin
        self.scale = 1.0
        self.amax_history = []

    def quantize(self, x):
        # Quantize with the scale derived from previous steps
        s = self.scale
        x_fp8 = (x * s).clamp(-self.fp8_max, self.fp8_max).to(torch.float8_e4m3fn)

        # Record this tensor's amax for future steps, keeping only the window
        self.amax_history = (self.amax_history + [x.abs().max().item()])[-self.history_len:]
        amax = max(self.amax_history)
        if amax > 0:  # an all-zero history leaves the scale unchanged
            self.scale = self.fp8_max / amax / 2 ** self.margin
        return x_fp8, s

    def dequantize(self, x_fp8, scale):
        return x_fp8.to(torch.float32) / scale''',
    "demo": """torch.manual_seed(0)
sc = DelayedScaler(history_len=4)
for step in range(4):
    x = torch.randn(256) * (step + 1)
    x_fp8, s = sc.quantize(x)
    x_hat = sc.dequantize(x_fp8, s)
    rel = ((x_hat - x).norm() / x.norm()).item()
    print(f'step={step}  scale_used={s:.4f}  next_scale={sc.scale:.4f}  rel_err={rel:.4f}')""",

}
