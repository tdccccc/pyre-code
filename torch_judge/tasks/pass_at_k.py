"""Unbiased pass@k task."""

TASK = {
    "title": "Unbiased pass@k",
    "title_zh": "pass@k 无偏估计",
    "difficulty": "Easy",
    "description_en": "Compute the unbiased pass@k estimator from the Codex paper (Chen et al., 2021), the standard metric for code and math benchmarks and for tracking RL training of reasoning models.\n\npass@k is the probability that at least one of $k$ samples solves a problem. Sampling exactly $k$ per problem gives a high-variance estimate, so instead $n \\ge k$ samples are drawn, $c$ of them are correct, and the probability is computed over all size-$k$ subsets of the $n$.\n\n**Signature:** `pass_at_k(num_samples: list[int], num_correct: list[int], k: int) -> float`\n\n**Parameters:**\n- `num_samples` — $n_j$, samples drawn for problem $j$ (one entry per problem, at least one problem)\n- `num_correct` — $c_j$, how many of them are correct ($0 \\le c_j \\le n_j$)\n- `k` — the budget, a positive int\n\n**Returns:** the mean over problems, as a Python float\n\n**Formula:**\n$$\n\\text{pass@}k = \\frac{1}{P}\\sum_{j=1}^{P}\\left[1 - \\frac{\\binom{n_j - c_j}{k}}{\\binom{n_j}{k}}\\right]\n$$\nThe term is $1$ when $n_j - c_j < k$ (every size-$k$ subset contains a correct sample).\n\n**Constraints:**\n- Raise `ValueError` if the two lists have different lengths, or if $k > n_j$ for any problem.\n- Must stay accurate for large $n$ (e.g. $n = 10000$, $k = 500$), where the binomial coefficients are far beyond the float range. The ratio equals $\\prod_{i=n-c+1}^{n}\\left(1 - \\frac{k}{i}\\right)$.",
    "description_zh": "计算 Codex 论文（Chen et al., 2021）提出的 pass@k 无偏估计。它是代码和数学评测的标准指标，也常用来跟踪推理模型的 RL 训练。\n\npass@k 是 $k$ 次采样中至少有一次做对的概率。每题只采 $k$ 次，估计的方差很大，所以改为采 $n \\ge k$ 次，其中 $c$ 次正确，再对这 $n$ 个样本的所有大小为 $k$ 的子集求概率。\n\n**签名:** `pass_at_k(num_samples: list[int], num_correct: list[int], k: int) -> float`\n\n**参数:**\n- `num_samples` — $n_j$，第 $j$ 题的采样次数（每题一项，至少一题）\n- `num_correct` — $c_j$，其中正确的次数（$0 \\le c_j \\le n_j$）\n- `k` — 预算，正整数\n\n**返回:** 所有题目的平均值，Python float\n\n**公式:**\n$$\n\\text{pass@}k = \\frac{1}{P}\\sum_{j=1}^{P}\\left[1 - \\frac{\\binom{n_j - c_j}{k}}{\\binom{n_j}{k}}\\right]\n$$\n当 $n_j - c_j < k$ 时这一项为 $1$（任意 $k$ 个样本里都必然有正确的）。\n\n**约束:**\n- 两个列表长度不同，或任一题 $k > n_j$ 时，抛出 `ValueError`。\n- $n$ 很大时（如 $n = 10000$、$k = 500$）也要准确，此时二项式系数远超浮点范围。比值等于 $\\prod_{i=n-c+1}^{n}\\left(1 - \\frac{k}{i}\\right)$。",
    "function_name": "pass_at_k",
    "hint": "check lengths and k <= n for every problem\nfor each (n, c):\n  if n - c < k: term = 1.0\n  else: term = 1 - prod(1 - k / i for i in range(n - c + 1, n + 1))\nreturn mean of the terms",
    "hint_zh": "检查长度一致，且每题 k <= n\n对每个 (n, c)：\n  若 n - c < k：term = 1.0\n  否则：term = 1 - prod(1 - k / i for i in range(n - c + 1, n + 1))\n返回各项的平均值",
    "tests": [
        {
            "name": "Hand-computed values and return type",
            "code": """
out = {fn}([10], [0], 1)
assert isinstance(out, float), f'Expected a Python float, got {type(out).__name__}'
assert out == 0.0, f'No correct samples: expected 0.0, got {out}'
assert {fn}([10], [10], 3) == 1.0, 'All samples correct should give 1.0'
assert abs({fn}([5], [2], 1) - 0.4) < 1e-12, 'pass@1 with n=5, c=2 is 2/5'
# n=4, c=1, k=2: 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
got = {fn}([4], [1], 2)
assert abs(got - 0.5) < 1e-12, f'n=4, c=1, k=2: expected 0.5, got {got}'
""",
        },
        {
            "name": "Matches the exact binomial formula",
            "code": """
import random
from fractions import Fraction
random.seed(0)
for _ in range(200):
    n = random.randint(1, 60)
    c = random.randint(0, n)
    k = random.randint(1, n)
    exact = 1 - Fraction(math.comb(n - c, k), math.comb(n, k))
    got = {fn}([n], [c], k)
    assert abs(got - float(exact)) < 1e-12, f'n={n}, c={c}, k={k}: expected {float(exact)}, got {got}'
""",
        },
        {
            "name": "Fewer than k wrong samples gives 1",
            "code": """
# n - c = 2 < k = 3: every 3-subset contains a correct sample
assert {fn}([5], [3], 3) == 1.0, 'n=5, c=3, k=3 should be exactly 1.0'
assert {fn}([8], [1], 8) == 1.0, 'k = n with at least one correct sample should be 1.0'
assert {fn}([8], [0], 8) == 0.0, 'k = n with no correct sample should be 0.0'
""",
        },
        {
            "name": "Mean over problems",
            "code": """
from fractions import Fraction
ns, cs, k = [10, 20, 7, 50], [3, 0, 7, 12], 4
terms = [1 - Fraction(math.comb(n - c, k), math.comb(n, k)) for n, c in zip(ns, cs)]
expected = float(sum(terms) / len(terms))
got = {fn}(ns, cs, k)
assert abs(got - expected) < 1e-12, f'Expected the mean over problems {expected}, got {got}'
assert abs({fn}(ns, cs, 1) - sum(c / n for n, c in zip(ns, cs)) / 4) < 1e-12, 'pass@1 is the mean of c/n'
""",
        },
        {
            "name": "Accurate for large n",
            "code": """
from fractions import Fraction
for n, c, k in [(10000, 37, 500), (2000, 1, 1999), (5000, 4000, 1000), (3000, 3, 1)]:
    exact = float(1 - Fraction(math.comb(n - c, k), math.comb(n, k)))
    got = {fn}([n], [c], k)
    assert abs(got - exact) < 1e-9, f'n={n}, c={c}, k={k}: expected {exact}, got {got}'
""",
        },
        {
            "name": "Invalid input raises ValueError",
            "code": """
cases = {
    'k larger than n': (([10, 3], [1, 0], 4),),
    'length mismatch': (([10, 10], [1], 2),),
}
for name, (args,) in cases.items():
    try:
        {fn}(*args)
    except ValueError:
        continue
    raise AssertionError(f'{name}: expected ValueError')
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
ns, cs = [10, 20, 30], [1, 2, 3]
{fn}(ns, cs, 5)
assert ns == [10, 20, 30] and cs == [1, 2, 3], 'Input lists were modified'
""",
        },
    ],
    "solution": '''def pass_at_k(num_samples, num_correct, k):
    if len(num_samples) != len(num_correct):
        raise ValueError("num_samples and num_correct must have the same length")
    total = 0.0
    for n, c in zip(num_samples, num_correct):
        if k > n:
            raise ValueError(f"k={k} is larger than n={n}")
        if n - c < k:
            total += 1.0
            continue
        # C(n-c, k) / C(n, k) as a product, so nothing overflows for large n
        prob_all_wrong = 1.0
        for i in range(n - c + 1, n + 1):
            prob_all_wrong *= 1.0 - k / i
        total += 1.0 - prob_all_wrong
    return total / len(num_samples)''',
    "demo": """# 200 samples per problem, as in the Codex paper
ns = [200, 200, 200, 200]
cs = [0, 3, 60, 190]
for k in [1, 5, 10, 50, 100]:
    print(f'pass@{k:<3} = {pass_at_k(ns, cs, k):.4f}')
# The naive estimate 1 - (1 - c/n)^k treats the k draws as with replacement and comes out lower
k = 10
naive = sum(1 - (1 - c / n) ** k for n, c in zip(ns, cs)) / len(ns)
print(f'naive pass@{k} = {naive:.4f}  vs unbiased {pass_at_k(ns, cs, k):.4f}')""",

}
