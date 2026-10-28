"""Minimum Path Sum (LeetCode 64)."""

import random


def _min_path(grid: list[list[int]]) -> int:
    # Full table with explicit first row and column; independent of the reference's rolling row
    m, n = len(grid), len(grid[0])
    cost = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                cost[i][j] = grid[i][j]
            elif i == 0:
                cost[i][j] = cost[i][j - 1] + grid[i][j]
            elif j == 0:
                cost[i][j] = cost[i - 1][j] + grid[i][j]
            else:
                cost[i][j] = grid[i][j] + min(cost[i - 1][j], cost[i][j - 1])
    return cost[-1][-1]


def _large_case():
    rng = random.Random(64)
    grid = [[rng.randint(0, 200) for _ in range(200)] for _ in range(200)]
    return {"args": [grid], "expected": _min_path(grid)}


def _random_case(rng):
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    return {"args": [[[rng.randint(0, 9) for _ in range(n)] for _ in range(m)]]}


TASK = {
    "title": "Minimum Path Sum",
    "title_zh": "最小路径和",
    "difficulty": "Medium",
    "leetcode_id": 64,
    "leetcode_slug": "minimum-path-sum",
    "category": "Multi-dimensional DP",
    "function_name": "minPathSum",
    "params": [("grid", "int[][]")],
    "returns": "int",
    "description_en": "Given an `m x n` grid of non-negative integers, find a route from the top-left cell to the bottom-right cell that makes the sum of its cell values as small as possible, and return that sum. Each step moves one cell either **right** or **down**. Both end cells are counted.\n\n**Example 1:** `grid = [[1,4,2],[5,1,1]]` → `7` (the route 1 → 4 → 1 → 1)\n\n**Example 2:** `grid = [[9],[3],[0]]` → `12` (the only route goes straight down)\n\n**Constraints:**\n- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 200$\n- $0 \\le \\texttt{grid[i][j]} \\le 200$",
    "description_zh": "给定一个由非负整数组成的 `m x n` 网格（`m` 行 `n` 列），找出一条从左上角到右下角的路线，使经过的格子数值之和最小，并返回这个最小和。每一步只能向**右**或向**下**走一格。起点和终点都要计入。\n\n**示例 1：** `grid = [[1,4,2],[5,1,1]]` → `7`（路线 1 → 4 → 1 → 1）\n\n**示例 2：** `grid = [[9],[3],[0]]` → `12`（唯一的路线是一直向下）\n\n**约束：**\n- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 200$\n- $0 \\le \\texttt{grid[i][j]} \\le 200$",
    "hint": "Let best[i][j] be the cheapest sum on a route to cell (i, j). It is grid[i][j] plus the cheaper of the cell above and the cell to the left, where a neighbour that is off the grid does not exist.",
    "hint_zh": "设 best[i][j] 为到达格子 (i, j) 的最小路线和。它等于 grid[i][j] 加上上方和左方两者中较小的那个；网格外的邻居不存在，不能参与比较。",
    "cases": [
        {"args": [[[1, 4, 2], [5, 1, 1]]], "expected": 7},
        {"args": [[[9], [3], [0]]], "expected": 12},
        {"args": [[[0]]], "expected": 0},
        {"args": [[[1, 2, 3]]], "expected": 6},
        {"args": [[[1], [2], [3]]], "expected": 6},
        {"args": [[[1, 3], [1, 5], [4, 2]]], "expected": 8},
        # Off-grid neighbours must not count as 0: a bad edge would make this 1, not 8
        {"args": [[[7, 1]]], "expected": 8},
        {"args": [[[1, 2], [3, 4]]], "expected": 7},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        # cost[j]: cheapest sum reaching column j of the current row
        cost = [0] * n
        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    cost[j] = grid[i][j]
                elif i == 0:
                    cost[j] = cost[j - 1] + grid[i][j]
                elif j == 0:
                    cost[j] = cost[j] + grid[i][j]
                else:
                    # cost[j] is still the cell above, cost[j - 1] already the cell to the left
                    cost[j] = grid[i][j] + min(cost[j], cost[j - 1])
        return cost[-1]''',
    "explanation": "The same table as Unique Paths, with minimum and sums instead of counts: `cost[i][j] = grid[i][j] + min(cost[i-1][j], cost[i][j-1])`. Cells in the first row or column have only one predecessor, so the code handles them separately rather than treating the missing neighbour as a value. The table is kept as a single row of `n` values, updated left to right.\n\nTime O(m · n), space O(n).",
    "explanation_zh": "与不同路径相同的表格，只是把计数换成求最小和：`cost[i][j] = grid[i][j] + min(cost[i-1][j], cost[i][j-1])`。第一行和第一列的格子只有一个前驱，所以单独处理，而不是把不存在的邻居当成某个值。表格压缩成一行 `n` 个值，从左到右更新。\n\n时间 O(m · n)，空间 O(n)。",
}
