"""Coin Change (LeetCode 322)."""

import random


def _min_coins(coins: list[int], amount: int) -> int:
    inf = amount + 1
    dp = [0] + [inf] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] <= amount else -1


def _large_case():
    rng = random.Random(322)
    coins = rng.sample(range(7, 500), 12)
    amount = 10000
    return {"args": [coins, amount], "expected": _min_coins(coins, amount)}


TASK = {
    "title": "Coin Change",
    "title_zh": "零钱兑换",
    "difficulty": "Medium",
    "leetcode_id": 322,
    "leetcode_slug": "coin-change",
    "category": "Dynamic Programming",
    "function_name": "coinChange",
    "params": [("coins", "int[]"), ("amount", "int")],
    "returns": "int",
    "description_en": "You have coins of the denominations in `coins`, with an unlimited supply of each. Return the fewest coins that add up to exactly `amount`, or `-1` if no combination does.\n\n**Example 1:** `coins = [1,3,4], amount = 6` → `2` (3 + 3; greedy 4 + 1 + 1 uses three)\n\n**Example 2:** `coins = [2], amount = 3` → `-1`\n\n**Constraints:**\n- `1 <= coins.length <= 12`\n- `1 <= coins[i] <= 2^31 - 1`\n- `0 <= amount <= 10^4`",
    "description_zh": "你有面额为 `coins` 中各值的硬币，每种数量不限。返回凑出恰好 `amount` 所需的最少硬币数；如果凑不出，返回 `-1`。\n\n**示例 1：** `coins = [1,3,4], amount = 6` → `2`（3 + 3；贪心的 4 + 1 + 1 要用三枚）\n\n**示例 2：** `coins = [2], amount = 3` → `-1`\n\n**约束：**\n- `1 <= coins.length <= 12`\n- `1 <= coins[i] <= 2^31 - 1`\n- `0 <= amount <= 10^4`",
    "hint": "Greedy fails (see example 1). Let dp[a] be the fewest coins for amount a: dp[0] = 0, and dp[a] = 1 + min(dp[a - c]) over coins c <= a.",
    "hint_zh": "贪心不对（见示例 1）。设 dp[a] 为凑出金额 a 的最少硬币数：dp[0] = 0，dp[a] = 1 + min(dp[a - c])，c 取所有不超过 a 的面额。",
    "cases": [
        {"args": [[1, 3, 4], 6], "expected": 2},
        {"args": [[2], 3], "expected": -1},
        {"args": [[1], 0], "expected": 0},
        {"args": [[2, 5, 10, 1], 27], "expected": 4},
        {"args": [[7, 11], 25], "expected": 3},
        {"args": [[2147483647], 2], "expected": -1},
    ],
    "large_case": _large_case,
    "solution": '''class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        inf = amount + 1  # more coins than any real answer needs
        dp = [0] + [inf] * amount
        for a in range(1, amount + 1):
            for c in coins:
                if c <= a and dp[a - c] + 1 < dp[a]:
                    dp[a] = dp[a - c] + 1
        return dp[amount] if dp[amount] <= amount else -1''',
    "explanation": "Complete-knapsack style DP. `dp[a]` is the fewest coins summing to `a`, with `dp[0] = 0`. Any optimal way to make `a` ends with some coin `c`, leaving an optimal way to make `a - c`, so `dp[a] = 1 + min(dp[a - c])`. `amount + 1` serves as infinity because no answer can use more than `amount` coins.\n\nTime O(amount × len(coins)), space O(amount).",
    "explanation_zh": "完全背包式的动态规划。`dp[a]` 表示凑出 `a` 的最少硬币数，`dp[0] = 0`。凑出 `a` 的最优方案一定以某枚硬币 `c` 结尾，剩下的部分正好是凑出 `a - c` 的最优方案，所以 `dp[a] = 1 + min(dp[a - c])`。用 `amount + 1` 当作无穷大，因为任何答案都不会超过 `amount` 枚。\n\n时间 O(amount × len(coins))，空间 O(amount)。",
}
