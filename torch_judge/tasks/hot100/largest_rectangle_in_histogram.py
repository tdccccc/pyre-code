"""Largest Rectangle in Histogram (LeetCode 84)."""

import random


def _largest(heights: list) -> int:
    # For each bar, the nearest shorter bar on each side, found with jump pointers. The bar
    # then spans everything strictly between those two shorter bars
    n = len(heights)
    left, right = [-1] * n, [n] * n
    for i in range(n):
        j = i - 1
        while j >= 0 and heights[j] >= heights[i]:
            j = left[j]
        left[i] = j
    for i in range(n - 1, -1, -1):
        j = i + 1
        while j < n and heights[j] >= heights[i]:
            j = right[j]
        right[i] = j
    return max(heights[i] * (right[i] - left[i] - 1) for i in range(n))


def _large_case():
    # 10^5 bars, the constraint limit. Sorted heights make every bar's rightward extension run
    # to the end of the histogram: about 5·10^9 steps for a naive expand-from-each-bar search.
    # A stack handles this in linear time
    rng = random.Random(84)
    heights = sorted(rng.randint(1, 10**4) for _ in range(10**5))
    return {"args": [heights], "expected": _largest(heights)}


def _random_case(rng):
    return {"args": [[rng.randint(1, 6) for _ in range(rng.randint(1, 8))]]}


TASK = {
    "title": "Largest Rectangle in Histogram",
    "title_zh": "柱状图中最大的矩形",
    "difficulty": "Hard",
    "leetcode_id": 84,
    "leetcode_slug": "largest-rectangle-in-histogram",
    "category": "Stack",
    "function_name": "largestRectangleArea",
    "params": [("heights", "int[]")],
    "returns": "int",
    "description_en": "A histogram is drawn with bars of width `1`, where `heights[i]` is the height of bar `i`. Return the area of the largest rectangle that fits entirely inside the histogram. The rectangle is made of consecutive bars, and its height is the height of the shortest bar it covers.\n\n**Example 1:** `heights = [1,3,3,1,2]` → `6` (the two bars of height 3)\n\n**Example 2:** `heights = [2,7]` → `7`\n\n**Constraints:**\n- $1 \\le \\texttt{heights.length} \\le 10^{5}$\n- $1 \\le \\texttt{heights[i]} \\le 10^{4}$",
    "description_zh": "柱状图由宽度为 `1` 的柱子组成，`heights[i]` 是第 `i` 根柱子的高度。返回柱状图中能完全包含在其内的最大矩形面积。矩形由相邻的柱子组成，它的高度是所覆盖的柱子中最矮的那根的高度。\n\n**示例 1：** `heights = [1,3,3,1,2]` → `6`（两根高度为 3 的柱子）\n\n**示例 2：** `heights = [2,7]` → `7`\n\n**约束：**\n- $1 \\le \\texttt{heights.length} \\le 10^{5}$\n- $1 \\le \\texttt{heights[i]} \\le 10^{4}$",
    "hint": "For each bar, find how far it can stretch left and right before meeting a shorter bar. A stack of bars with increasing heights finds those boundaries in one pass. Remember to flush the stack at the end.",
    "hint_zh": "对每根柱子，找出它向左、向右能延伸到哪里，直到遇到更矮的柱子。用一个保存递增高度柱子的栈，一次遍历就能找到这些边界。别忘了在结尾把栈清空处理。",
    "cases": [
        {"args": [[1, 3, 3, 1, 2]], "expected": 6},
        {"args": [[2, 7]], "expected": 7},
        {"args": [[5]], "expected": 5},
        # The tallest bar touches the end: a stack without a final flush never measures it
        {"args": [[2, 4]], "expected": 4},
        {"args": [[6, 2, 5, 4, 5, 1, 6]], "expected": 12},
        # Equal bars form one rectangle
        {"args": [[3, 3, 3]], "expected": 9},
        {"args": [[4, 3, 2, 1]], "expected": 6},
        {"args": [[1, 2, 3, 4, 5]], "expected": 9},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []  # indices of bars with increasing heights
        best = 0
        for i in range(len(heights) + 1):
            h = heights[i] if i < len(heights) else 0  # a final 0 flushes the stack
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0  # just after the nearest shorter bar
                best = max(best, heights[top] * (i - left))
            stack.append(i)
        return best''',
    "explanation": "Monotonic stack of bar indices with strictly increasing heights. When a bar shorter than the stack top arrives, the top bar cannot extend any further right, so its rectangle is fixed: its height times the distance from the bar below it in the stack to the current index. Equal bars are popped too, so each rectangle is measured once. A final sentinel of height 0 empties the stack.\n\nTime O(n), space O(n).",
    "explanation_zh": "用单调栈保存柱子下标，栈中柱子高度严格递增。当遇到比栈顶更矮的柱子时，栈顶柱子无法再向右延伸，它对应的矩形就确定了：高度乘以从栈中它下面那根柱子到当前下标之间的距离。等高的柱子也会被弹出，因此每个矩形只计算一次。末尾加一个高度为 0 的哨兵，把栈清空。\n\n时间 O(n)，空间 O(n)。",
}
