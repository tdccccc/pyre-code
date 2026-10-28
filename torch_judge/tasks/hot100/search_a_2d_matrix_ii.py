"""Search a 2D Matrix II (LeetCode 240)."""

import random


def _large_case():
    # 300 x 300 grid built the same way as the random cases: every row and column is
    # non-decreasing. The target is drawn from a range that holds both present and absent values
    rng = random.Random(240)
    m = n = 300
    grid = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            low = max(grid[i - 1][j] if i else -5, grid[i][j - 1] if j else -5)
            grid[i][j] = low + rng.randint(0, 3)
    present = {v for row in grid for v in row}
    target = rng.randint(-5, max(present) + 3)
    return {"args": [grid, target], "expected": target in present}


def _random_case(rng):
    # Every cell is at least its upper and left neighbours, so rows and columns never decrease.
    # Wide steps make each row end well above the next row's start, which catches a row-major
    # binary search (the mistake in scripts/hot100_mutants) on most seeds
    m, n = rng.randint(1, 6), rng.randint(1, 6)
    grid = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            low = max(grid[i - 1][j] if i else -5, grid[i][j - 1] if j else -5)
            grid[i][j] = low + rng.randint(0, 6)
    flat = [v for row in grid for v in row]
    target = rng.choice(flat) if rng.random() < 0.5 else rng.randint(-5, 15)
    return {"args": [grid, target]}


TASK = {
    "title": "Search a 2D Matrix II",
    "title_zh": "搜索二维矩阵 II",
    "difficulty": "Medium",
    "leetcode_id": 240,
    "leetcode_slug": "search-a-2d-matrix-ii",
    "category": "Matrix",
    "function_name": "searchMatrix",
    "params": [("matrix", "int[][]"), ("target", "int")],
    "returns": "bool",
    "description_en": "Given an `m x n` integer matrix in which every row and every column is sorted in non-decreasing order, decide whether `target` appears anywhere in the matrix. Return `true` or `false`.\n\nNote that a row's first value need not be larger than the previous row's last value.\n\n**Example 1:** `matrix = [[2,3,9],[4,6,10],[8,12,15]], target = 6` → `true`\n\n**Example 2:** `matrix = [[2,3,9],[4,6,10],[8,12,15]], target = 11` → `false`\n\n**Constraints:**\n- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 300$\n- $-10^{9} \\le \\texttt{matrix[i][j]},\\ \\texttt{target} \\le 10^{9}$",
    "description_zh": "给定 `m x n` 的整数矩阵，其中每一行和每一列都按非递减顺序排列（行从左到右，列从上到下）。判断 `target` 是否出现在矩阵中，返回 `true` 或 `false`。\n\n注意：上一行的最后一个数不一定小于下一行的第一个数。\n\n**示例 1：** `matrix = [[2,3,9],[4,6,10],[8,12,15]], target = 6` → `true`\n\n**示例 2：** `matrix = [[2,3,9],[4,6,10],[8,12,15]], target = 11` → `false`\n\n**约束：**\n- $1 \\le \\texttt{m},\\ \\texttt{n} \\le 300$\n- $-10^{9} \\le \\texttt{matrix[i][j]},\\ \\texttt{target} \\le 10^{9}$",
    "hint": "Start at the top-right corner. That value is the largest in its row and the smallest in its column, so each comparison can rule out a whole row or a whole column.",
    "hint_zh": "从右上角出发。这个值是它所在行的最大值、所在列的最小值，所以每次比较都能排除一整行或一整列。",
    "cases": [
        {"args": [[[2, 3, 9], [4, 6, 10], [8, 12, 15]], 6], "expected": True},
        {"args": [[[2, 3, 9], [4, 6, 10], [8, 12, 15]], 11], "expected": False},
        {"args": [[[-5]], -5], "expected": True},
        # Flattening the grid into one sorted list and binary searching it misses this one:
        # the end of row 0 (5) is larger than the start of row 1 (2)
        {"args": [[[1, 5], [2, 6]], 2], "expected": True},
        {"args": [[[2, 3], [4, 6]], 1], "expected": False},
        {"args": [[[1, 2], [3, 4]], 4], "expected": True},
        {"args": [[[1, 2, 3, 4, 5]], 4], "expected": True},
        {"args": [[[1], [2], [3]], 5], "expected": False},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row, col = 0, len(matrix[0]) - 1  # start at the top-right corner
        while row < len(matrix) and col >= 0:
            v = matrix[row][col]
            if v == target:
                return True
            if v > target:
                col -= 1  # everything below v in this column is at least v, too big
            else:
                row += 1  # everything left of v in this row is at most v, too small
        return False''',
    "explanation": "Staircase search from the top-right corner. That value is the largest in its row and the smallest in its column. If it is bigger than `target`, its whole column is too big, so drop the column. If it is smaller, its whole row is too small, so drop the row. Each step removes a row or a column, and the search stops when the target is found or the walk leaves the grid.\n\nTime O(m + n), space O(1).",
    "explanation_zh": "从右上角开始的阶梯式搜索。这个值是它所在行的最大值、所在列的最小值。如果它比 `target` 大，那么它所在的整列都不小于它，也都比 `target` 大，于是排除这一列；如果它比 `target` 小，那么它所在的整行都不大于它，也都比 `target` 小，于是排除这一行。每一步都排除一行或一列，直到找到 `target` 或者走出矩阵。\n\n时间 O(m + n)，空间 O(1)。",
}
