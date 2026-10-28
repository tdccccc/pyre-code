"""Trapping Rain Water (LeetCode 42)."""

import random


def _trapped(height: list[int]) -> int:
    # Independent method: prefix maxima from the left and suffix maxima from the right
    n = len(height)
    left_max = [0] * n
    best = 0
    for i, h in enumerate(height):
        best = max(best, h)
        left_max[i] = best
    best = 0
    total = 0
    for i in range(n - 1, -1, -1):
        best = max(best, height[i])
        total += min(left_max[i], best) - height[i]
    return total


def _large_case():
    # 2·10⁴ bars at the constraint limit. Scanning both sides from every cell is about 4·10⁸
    # steps, far beyond 3 s; the two-pointer scan visits each bar once.
    rng = random.Random(42)
    height = [rng.randint(0, 10**5) for _ in range(20000)]
    return {"args": [height], "expected": _trapped(height)}


def _random_case(rng):
    return {"args": [[rng.randint(0, 4) for _ in range(rng.randint(1, 12))]]}


TASK = {
    "title": "Trapping Rain Water",
    "title_zh": "接雨水",
    "difficulty": "Hard",
    "leetcode_id": 42,
    "leetcode_slug": "trapping-rain-water",
    "category": "Two Pointers",
    "function_name": "trap",
    "params": [("height", "int[]")],
    "returns": "int",
    "description_en": "You are given `n` non-negative integers `height`. Each one is the height of a bar of width 1, and the bars stand side by side. After rain, water collects in the dips between taller bars. Return the total number of units of water the bars trap.\n\n**Example 1:** `height = [0,2,0,3,1,0,4]` → `7`\n\n**Example 2:** `height = [3,0,0,2,0,4]` → `10`\n\n**Constraints:**\n- $1 \\le \\texttt{n} \\le 2 \\times 10^{4}$\n- $0 \\le \\texttt{height[i]} \\le 10^{5}$",
    "description_zh": "给定 `n` 个非负整数 `height`，每个数表示一根宽度为 1 的柱子的高度，柱子紧挨着排列。下雨后，水会积在较高柱子之间的凹处。返回这些柱子总共能接住多少单位的水。\n\n**示例 1：** `height = [0,2,0,3,1,0,4]` → `7`\n\n**示例 2：** `height = [3,0,0,2,0,4]` → `10`\n\n**约束：**\n- $1 \\le \\texttt{n} \\le 2 \\times 10^{4}$\n- $0 \\le \\texttt{height[i]} \\le 10^{5}$",
    "hint": "The water above a cell is the lower of the tallest bars to its left and right, minus its own height. Keep running maxima from both ends and always settle the cell on the lower side.",
    "hint_zh": "某个位置上方的水位，是它左右两侧最高柱子中较矮的那个，减去它自己的高度。从两端维护左右最大值，每次结算较矮一侧的位置。",
    "cases": [
        {"args": [[0, 2, 0, 3, 1, 0, 4]], "expected": 7},
        {"args": [[3, 0, 0, 2, 0, 4]], "expected": 10},
        {"args": [[5]], "expected": 0},
        # Index 3 (height 3) is held up by the 4 further left, not by its neighbours (0 and 2)
        {"args": [[4, 2, 0, 3, 2, 5]], "expected": 9},
        {"args": [[2, 0, 2]], "expected": 2},
        {"args": [[5, 4, 3, 2, 1]], "expected": 0},
        # Two tall walls at the ends and a flat floor: 6 cells, each holding 6 units
        {"args": [[6, 0, 0, 0, 0, 0, 0, 6]], "expected": 36},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def trap(self, height: List[int]) -> int:
        lo, hi = 0, len(height) - 1
        left_max = right_max = 0
        water = 0
        while lo < hi:
            # Settle the side with the lower wall: the other side already has a taller one
            if height[lo] < height[hi]:
                left_max = max(left_max, height[lo])
                water += left_max - height[lo]
                lo += 1
            else:
                right_max = max(right_max, height[hi])
                water += right_max - height[hi]
                hi -= 1
        return water''',
    "explanation": "The water above index `i` is `min(max_left, max_right) - height[i]`. Two pointers walk inward from both ends, keeping the highest bar seen on each side. Whichever side has the lower current wall is settled first: the opposite side already holds a wall at least that tall, so the water above that cell is set by the running maximum on its own side. Each cell is settled exactly once.\n\nTime O(n), space O(1).",
    "explanation_zh": "位置 `i` 上方的水位是 `min(左侧最高墙, 右侧最高墙) - height[i]`。双指针从两端向中间移动，分别记录两侧见过的最高柱子。两端中较矮的那一侧先结算：另一侧至少有一根不矮于它的柱子，所以这个位置的水位只由它自己一侧的运行最大值决定。每个位置只结算一次。\n\n时间 O(n)，空间 O(1)。",
}
