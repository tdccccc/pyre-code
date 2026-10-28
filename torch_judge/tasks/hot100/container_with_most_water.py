"""Container With Most Water (LeetCode 11)."""

import random


def _most_water(height: list[int]) -> int:
    # Independent method: for each wall height m, the outermost bars at least m tall give the
    # widest container whose shorter wall is at least m; the best such m*width is the answer
    first, last = {}, {}
    for i, h in enumerate(height):
        first.setdefault(h, i)
        last[h] = i
    best = 0
    left = right = None
    for m in range(max(height), -1, -1):
        if m in first:
            left = first[m] if left is None else min(left, first[m])
            right = last[m] if right is None else max(right, last[m])
        if left is not None:
            best = max(best, m * (right - left))
    return best


def _large_case():
    # 10^5 bars at the constraint limit. Trying every pair costs about 5·10⁹ checks, far beyond
    # 3 s, while the two-pointer scan takes 10^5 steps.
    rng = random.Random(11)
    height = [rng.randint(0, 10**4) for _ in range(100000)]
    return {"args": [height], "expected": _most_water(height)}


def _random_case(rng):
    return {"args": [[rng.randint(0, 6) for _ in range(rng.randint(2, 8))]]}


TASK = {
    "title": "Container With Most Water",
    "title_zh": "盛最多水的容器",
    "difficulty": "Medium",
    "leetcode_id": 11,
    "leetcode_slug": "container-with-most-water",
    "category": "Two Pointers",
    "function_name": "maxArea",
    "params": [("height", "int[]")],
    "returns": "int",
    "description_en": "You are given an integer array `height` of length `n`. Vertical line `i` runs from `(i, 0)` up to `(i, height[i])`. Choose two of these lines. Together with the x-axis they form a container, and the water it holds is its width times the height of the shorter line. Return the largest amount of water any such container can hold.\n\n**Example 1:** `height = [2,7,1,5]` → `10` (the lines at indices 1 and 3: width 2, shorter wall 5)\n\n**Example 2:** `height = [1,1]` → `1`\n\n**Constraints:**\n- `2 <= n <= 10^5`\n- `0 <= height[i] <= 10^4`",
    "description_zh": "给定长度为 `n` 的整数数组 `height`。第 `i` 条竖线从 `(i, 0)` 画到 `(i, height[i])`。从中选出两条线，与 x 轴一起围成一个容器，它能装的水量等于两线的间距乘以较矮那条线的高度。返回所有容器中能装的最大水量。\n\n**示例 1：** `height = [2,7,1,5]` → `10`（下标 1 与 3 的两条线：间距 2，较矮的墙高 5）\n\n**示例 2：** `height = [1,1]` → `1`\n\n**约束：**\n- `2 <= n <= 10^5`\n- `0 <= height[i] <= 10^4`",
    "hint": "The shorter wall limits the area. Start with the two ends and always move the shorter wall inward: the width shrinks, and the taller wall cannot make up for it.",
    "hint_zh": "较矮的墙决定了面积上限。从两端开始，每次把较矮的墙向内移：宽度变小，而较高的墙也补偿不了这一点。",
    "cases": [
        {"args": [[2, 7, 1, 5]], "expected": 10},
        {"args": [[1, 1]], "expected": 1},
        # Moving the taller wall (8) inward drops the pair of 9 and 8, which is the best one
        {"args": [[1, 9, 2, 8]], "expected": 16},
        {"args": [[4, 3, 2, 1, 4]], "expected": 16},
        {"args": [[1, 2, 1]], "expected": 2},
        {"args": [[5, 0, 0, 0, 0, 5]], "expected": 25},
        {"args": [[0, 0, 0]], "expected": 0},
        {"args": [[2, 3, 10, 5, 7, 8, 9]], "expected": 36},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def maxArea(self, height: List[int]) -> int:
        lo, hi = 0, len(height) - 1
        best = 0
        while lo < hi:
            best = max(best, min(height[lo], height[hi]) * (hi - lo))
            # The shorter wall cannot pair with anything inside to do better, so drop it
            if height[lo] < height[hi]:
                lo += 1
            else:
                hi -= 1
        return best''',
    "explanation": "Two pointers start at the widest pair. The area is `min(height[l], height[r]) * (r - l)`. If `height[l] < height[r]`, the left wall is the bottleneck. Any partner `j` strictly between `l` and `r` gives a narrower container whose shorter wall is at most `height[l]`, so none can beat the pair `(l, r)` already measured. Discard `l` and move it right. Otherwise discard `r` the same way. Every pair that could beat the best is still in play when its wall is discarded, so the scan finds the maximum.\n\nTime O(n), space O(1).",
    "explanation_zh": "双指针从最宽的一对开始。面积为 `min(height[l], height[r]) * (r - l)`。若 `height[l] < height[r]`，左墙是瓶颈：与 `l`、`r` 之间任一竖线 `j` 配对，宽度更小，且较矮的墙不会高于 `height[l]`，都不可能超过已经算过的 `(l, r)`。于是丢弃 `l`，左指针右移；否则同理丢弃 `r`。每次丢弃时，可能更优的组合都已经被考虑过，所以最终结果是最大值。\n\n时间 O(n)，空间 O(1)。",
}
