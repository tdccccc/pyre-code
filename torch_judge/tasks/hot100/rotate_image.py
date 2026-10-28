"""Rotate Image (LeetCode 48)."""

import random


def _rotated_cw(matrix):
    # Row i of the result is column i of the input, read from the bottom up
    n = len(matrix)
    return [[matrix[n - 1 - j][i] for j in range(n)] for i in range(n)]


def _large_case():
    # The constraints cap the grid at 20 x 20, so no approach can time out here. This case
    # checks a full-size grid instead
    rng = random.Random(48)
    matrix = [[rng.randint(-1000, 1000) for _ in range(20)] for _ in range(20)]
    return {"args": [matrix], "expected": _rotated_cw(matrix)}


def _random_case(rng):
    n = rng.randint(1, 4)
    return {"args": [[[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]]}


TASK = {
    "title": "Rotate Image",
    "title_zh": "旋转图像",
    "difficulty": "Medium",
    "leetcode_id": 48,
    "leetcode_slug": "rotate-image",
    "category": "Matrix",
    "function_name": "rotate",
    "params": [("matrix", "int[][]")],
    "returns": "void",
    "inplace": "matrix",
    "description_en": "Given an `n x n` matrix, rotate it 90 degrees clockwise **in place**. Do not return anything.\n\n**Example 1:** `matrix = [[7,8,9],[4,5,6],[1,2,3]]` → `[[1,4,7],[2,5,8],[3,6,9]]`\n\n**Example 2:** `matrix = [[1,2],[3,4]]` → `[[3,1],[4,2]]`\n\n**Constraints:**\n- $1 \\le \\texttt{n} \\le 20$\n- $-1000 \\le \\texttt{matrix[i][j]} \\le 1000$",
    "description_zh": "给定一个 `n x n` 的矩阵，把它顺时针旋转 90 度，并且**原地**完成。不要返回任何值。\n\n**示例 1：** `matrix = [[7,8,9],[4,5,6],[1,2,3]]` → `[[1,4,7],[2,5,8],[3,6,9]]`\n\n**示例 2：** `matrix = [[1,2],[3,4]]` → `[[3,1],[4,2]]`\n\n**约束：**\n- $1 \\le \\texttt{n} \\le 20$\n- $-1000 \\le \\texttt{matrix[i][j]} \\le 1000$",
    "hint": "Two simple steps make a 90-degree clockwise turn: swap across the main diagonal, then reverse every row. Each step can be done in place.",
    "hint_zh": "两个简单的步骤组合起来就是顺时针旋转 90 度：先沿主对角线交换，再把每一行反转。两步都可以原地完成。",
    "cases": [
        {"args": [[[7, 8, 9], [4, 5, 6], [1, 2, 3]]], "expected": [[1, 4, 7], [2, 5, 8], [3, 6, 9]]},
        {"args": [[[1, 2], [3, 4]]], "expected": [[3, 1], [4, 2]]},
        {"args": [[[5]]], "expected": [[5]]},
        {"args": [[[-1, 0], [2, -3]]], "expected": [[2, -1], [-3, 0]]},
        {"args": [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]],
         "expected": [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        # Transpose: swap across the main diagonal
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        # Reverse each row: together with the transpose this is a clockwise turn
        for row in matrix:
            row.reverse()''',
    "explanation": "Transpose the matrix (swap `matrix[i][j]` with `matrix[j][i]`), then reverse every row. After the transpose, the first row is the original first column read top to bottom. Reversing it reads that column bottom to top, which is exactly the first row of a clockwise rotation, and the same holds for every other row.\n\nTime O(n²), space O(1).",
    "explanation_zh": "先转置矩阵（交换 `matrix[i][j]` 和 `matrix[j][i]`），再把每一行反转。转置后，第一行就是原来第一列从上到下的读数；反转之后变成从下到上读，正好是顺时针旋转后的第一行，其他行同理。\n\n时间 O(n²)，空间 O(1)。",
}
