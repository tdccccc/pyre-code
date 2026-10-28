"""Unique Paths (LeetCode 62)."""

import random


def _paths(m: int, n: int) -> int:
    # Full table, so the helper is independent of the reference's single rolling row
    grid = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] = grid[i - 1][j] + grid[i][j - 1]
    return grid[-1][-1]


def _large_case():
    # The largest square-ish grid whose answer stays within the 2 * 10^9 guarantee:
    # C(33, 16) ≈ 1.2 * 10^9 routes. Recursion without memoisation visits every one of them
    return {"args": [17, 18], "expected": _paths(17, 18)}


def _random_case(rng):
    return {"args": [rng.randint(1, 7), rng.randint(1, 7)]}


TASK = {
    "title": "Unique Paths",
    "title_zh": "不同路径",
    "difficulty": "Medium",
    "leetcode_id": 62,
    "leetcode_slug": "unique-paths",
    "category": "Multi-dimensional DP",
    "function_name": "uniquePaths",
    "params": [("m", "int"), ("n", "int")],
    "returns": "int",
    "description_en": "A robot stands on the top-left cell of a grid with `m` rows and `n` columns, and wants to reach the bottom-right cell. Each step it moves one cell either **right** or **down**. Return how many different routes lead from the start to the finish.\n\n**Example 1:** `m = 2, n = 4` → `4` (the single down-step can be taken after 0, 1, 2 or 3 right-steps)\n\n**Example 2:** `m = 1, n = 6` → `1` (the only route runs straight along the row)\n\n**Constraints:**\n- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 100$\n- The answer is at most $2 \\times 10^{9}$",
    "description_zh": "一个机器人站在有 `m` 行 `n` 列的网格左上角，想走到右下角。每一步它只能向**右**或向**下**走一格。返回从起点到终点共有多少条不同的路线。\n\n**示例 1：** `m = 2, n = 4` → `4`（唯一一次向下的移动，可以在向右走 0、1、2 或 3 步之后进行）\n\n**示例 2：** `m = 1, n = 6` → `1`（唯一的路线是沿着这一行一直向右）\n\n**约束：**\n- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 100$\n- 答案不超过 $2 \\times 10^{9}$",
    "hint": "The number of ways to reach a cell is the number of ways to reach the cell above it plus the number to its left. The first row and first column are all 1.",
    "hint_zh": "到达某个格子的路线数等于到达其上方格子的路线数加上到达其左方格子的路线数。第一行和第一列都是 1。",
    "cases": [
        {"args": [2, 4], "expected": 4},
        {"args": [1, 6], "expected": 1},
        {"args": [1, 1], "expected": 1},
        {"args": [3, 3], "expected": 6},
        {"args": [4, 4], "expected": 20},
        {"args": [5, 1], "expected": 1},
        {"args": [7, 3], "expected": 28},
        {"args": [10, 10], "expected": 48620},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # ways[j]: routes to column j of the current row; the first row has one route per cell
        ways = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                # ways[j] still holds the cell above; ways[j - 1] already holds the cell to the left
                ways[j] += ways[j - 1]
        return ways[-1]''',
    "explanation": "A cell can only be reached from above or from the left, so `paths[i][j] = paths[i-1][j] + paths[i][j-1]`, and every cell in the first row and column has exactly one path. One row of the table is enough: while updating left to right, `ways[j]` still holds the value from the row above, and `ways[j - 1]` has already been updated for the current row.\n\nAlternatively the answer is the binomial coefficient `C(m + n - 2, m - 1)`.\n\nTime O(m · n), space O(n).",
    "explanation_zh": "每个格子只能从上方或左方到达，所以 `paths[i][j] = paths[i-1][j] + paths[i][j-1]`，第一行和第一列的每个格子都只有一条路线。只用一行的数组就够了：从左到右更新时，`ways[j]` 仍保存着上一行的值，而 `ways[j - 1]` 已经更新为当前行的值。\n\n另外，答案也可以直接写成组合数 `C(m + n - 2, m - 1)`。\n\n时间 O(m · n)，空间 O(n)。",
}
