# Mistake: takes the signed max instead of the max absolute value, so all-negative inputs get the wrong scale
class DelayedScaler:
    def __init__(self, history_len=16, fp8_max=448.0, margin=0):
        self.history_len = history_len
        self.fp8_max = fp8_max
        self.margin = margin
        self.scale = 1.0
        self.amax_history = []

    def quantize(self, x):
        s = self.scale
        x_fp8 = (x * s).clamp(-self.fp8_max, self.fp8_max).to(torch.float8_e4m3fn)

        self.amax_history = (self.amax_history + [x.max().item()])[-self.history_len:]
        amax = max(self.amax_history)
        if amax > 0:
            self.scale = self.fp8_max / amax / 2 ** self.margin
        return x_fp8, s

    def dequantize(self, x_fp8, scale):
        return x_fp8.to(torch.float32) / scale
