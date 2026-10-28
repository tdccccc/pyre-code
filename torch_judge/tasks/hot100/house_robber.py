"""House Robber (LeetCode 198)."""

import random


def _best_loot(nums: list[int]) -> int:
    # dp[i + 2] is the best total over houses 0..i (dp[0] and dp[1] are the empty prefix)
    dp = [0] * (len(nums) + 2)
    for i, x in enumerate(nums):
        dp[i + 2] = max(dp[i + 1], dp[i] + x)
    return dp[-1]


def _large_case():
    # 100 houses (the maximum). Trying every subset of houses is 2^100 choices, far too many
    rng = random.Random(198)
    nums = [rng.randint(0, 400) for _ in range(100)]
    return {"args": [nums], "expected": _best_loot(nums)}


def _random_case(rng):
    return {"args": [[rng.randint(0, 9) for _ in range(rng.randint(1, 8))]]}


TASK = {
    "title": "House Robber",
    "title_zh": "打家劫舍",
    "difficulty": "Medium",
    "leetcode_id": 198,
    "leetcode_slug": "house-robber",
    "category": "Dynamic Programming",
    "function_name": "rob",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "A street has houses in a row, and `nums[i]` is the cash hidden in house `i`. You may rob any set of houses, but never two houses that sit next to each other, because adjacent houses trigger one alarm. Return the largest total you can take.\n\n**Example 1:** `nums = [4,1,3,7]` → `11` (houses 0 and 3)\n\n**Example 2:** `nums = [2,9,1,1,5,2]` → `14` (houses 1 and 4)\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 100$\n- $0 \\le \\texttt{nums[i]} \\le 400$",
    "description_zh": "一条街上的房屋排成一排，`nums[i]` 为第 `i` 间屋子里藏的现金。你可以选择任意一组房屋去抢，但不能抢相邻的两间，因为相邻的屋子会共用同一套报警系统。返回能抢到的最大金额。\n\n**示例 1：** `nums = [4,1,3,7]` → `11`（抢下标 0 和 3 的屋子）\n\n**示例 2：** `nums = [2,9,1,1,5,2]` → `14`（抢下标 1 和 4 的屋子）\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 100$\n- $0 \\le \\texttt{nums[i]} \\le 400$",
    "hint": "For each house, the best total either skips it (keep the best up to the previous house) or robs it (add it to the best up to the house before the previous one).",
    "hint_zh": "对每间屋子，最优值要么是不抢它（沿用前一间的最优值），要么是抢它（加上前前一间的最优值）。",
    "cases": [
        {"args": [[4, 1, 3, 7]], "expected": 11},
        {"args": [[2, 9, 1, 1, 5, 2]], "expected": 14},
        {"args": [[0]], "expected": 0},
        {"args": [[7]], "expected": 7},
        {"args": [[5, 5]], "expected": 5},
        {"args": [[2, 1, 1, 2]], "expected": 4},
        # All even-indexed houses (10 + 1 = 11) and all odd-indexed ones (1 + 10 = 11) both lose
        # to houses 0 and 3 together (20)
        {"args": [[10, 1, 1, 10]], "expected": 20},
        {"args": [[3, 10, 3, 1, 2]], "expected": 12},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def rob(self, nums: List[int]) -> int:
        skip_last, best = 0, 0  # best total over houses before the previous one, and up to it
        for x in nums:
            # Either skip this house (keep best) or rob it on top of the best two houses back
            skip_last, best = best, max(best, skip_last + x)
        return best''',
    "explanation": "Let `best(i)` be the most you can take from houses `0..i`. House `i` is either skipped, giving `best(i - 1)`, or robbed, which forbids house `i - 1` and gives `best(i - 2) + nums[i]`. So `best(i) = max(best(i - 1), best(i - 2) + nums[i])`. Only the last two values are kept.\n\nTime O(n), space O(1).",
    "explanation_zh": "设 `best(i)` 为只从前 `i + 1` 间屋子（下标 `0..i`）中能抢到的最大金额。第 `i` 间屋子要么不抢，得到 `best(i - 1)`；要么抢，这时不能抢第 `i - 1` 间，得到 `best(i - 2) + nums[i]`。所以 `best(i) = max(best(i - 1), best(i - 2) + nums[i])`。只需保存最近两个值。\n\n时间 O(n)，空间 O(1)。",
}
