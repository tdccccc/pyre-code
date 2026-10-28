"""Perfect Squares (LeetCode 279)."""

import math
import random


def _least_squares(n: int) -> int:
    # best[m] is the fewest squares summing to m, built for each m from the smaller values
    best = [0]
    for m in range(1, n + 1):
        best.append(min(best[m - k * k] for k in range(1, math.isqrt(m) + 1)) + 1)
    return best[n]


def _large_case():
    # n near 10^4 (the maximum). Each target is checked against about 100 squares; the search
    # over all orderings of squares (exponential) is what times out
    rng = random.Random(279)
    n = rng.randint(9000, 10000)
    return {"args": [n], "expected": _least_squares(n)}


# Greedy (largest square first) is wrong for roughly 40% of n up to 300. The mistake only shows
# once the optimum needs several squares, which takes n well past 60, so the range goes to 300
def _random_case(rng):
    return {"args": [rng.randint(1, 300)]}


TASK = {
    "title": "Perfect Squares",
    "title_zh": "完全平方数",
    "difficulty": "Medium",
    "leetcode_id": 279,
    "leetcode_slug": "perfect-squares",
    "category": "Dynamic Programming",
    "function_name": "numSquares",
    "params": [("n", "int")],
    "returns": "int",
    "description_en": "A perfect square is an integer that is the square of an integer, such as `1`, `4`, `9`, and `16`. Given a positive integer `n`, return the fewest perfect squares whose sum is exactly `n`. The same square may be used more than once.\n\n**Example 1:** `n = 18` → `2` (9 + 9)\n\n**Example 2:** `n = 6` → `3` (4 + 1 + 1)\n\n**Constraints:**\n- `1 <= n <= 10^4`",
    "description_zh": "完全平方数是某个整数的平方，例如 `1`、`4`、`9`、`16`。给定正整数 `n`，返回和恰好等于 `n` 的完全平方数的最少个数。同一个平方数可以使用多次。\n\n**示例 1：** `n = 18` → `2`（9 + 9）\n\n**示例 2：** `n = 6` → `3`（4 + 1 + 1）\n\n**约束：**\n- `1 <= n <= 10^4`",
    "hint": "Let best[m] be the fewest squares that sum to m. Each m is best[m - s] + 1 minimised over every square s <= m. Taking the largest square each time is not always optimal.",
    "hint_zh": "设 best[m] 为和为 m 的完全平方数的最少个数。对每个 m，在所有不超过 m 的平方数 s 上取 best[m - s] + 1 的最小值。每次都取最大的平方数并不总是最优。",
    "cases": [
        {"args": [18], "expected": 2},
        {"args": [6], "expected": 3},
        {"args": [1], "expected": 1},
        {"args": [4], "expected": 1},
        # Greedy takes 9 + 1 + 1 + 1 (four squares), but 4 + 4 + 4 uses three
        {"args": [12], "expected": 3},
        {"args": [13], "expected": 2},
        {"args": [7], "expected": 4},
        {"args": [10000], "expected": 1},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def numSquares(self, n: int) -> int:
        squares = []
        k = 1
        while k * k <= n:
            squares.append(k * k)
            k += 1
        best = list(range(n + 1))  # n copies of 1 is always a valid (if poor) answer
        for m in range(1, n + 1):
            for s in squares:
                if s > m:
                    break
                best[m] = min(best[m], best[m - s] + 1)  # last square is s
        return best[n]''',
    "explanation": "Let `best[m]` be the fewest perfect squares summing to `m`. The last square used in an optimal sum for `m` is some square `s <= m`, and the rest of that sum is an optimal sum for `m - s`. So `best[m] = min(best[m - s] + 1)` over all squares `s <= m`. Only the first `√n` squares are needed.\n\nTime O(n·√n), space O(n).",
    "explanation_zh": "设 `best[m]` 为和为 `m` 的完全平方数的最少个数。和为 `m` 的最优方案中，最后一个平方数是某个不超过 `m` 的 `s`，其余部分一定是和为 `m - s` 的最优方案。因此 `best[m] = min(best[m - s] + 1)`，对所有不超过 `m` 的平方数 `s` 取最小。只需用到不超过 `n` 的平方数，共 √n 个。\n\n时间 O(n·√n)，空间 O(n)。",
}
