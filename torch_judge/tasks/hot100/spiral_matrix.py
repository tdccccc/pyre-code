"""Spiral Matrix (LeetCode 54)."""

import random


def _spiral(matrix):
    # Pop the top row, then rotate what is left a quarter turn counter-clockwise and repeat
    out, rows = [], [list(r) for r in matrix]
    while rows:
        out.extend(rows.pop(0))
        rows = [list(r) for r in zip(*rows)][::-1]
    return out


def _large_case():
    # The constraints cap the grid at 10 x 10, so no approach can time out here. This case
    # checks a full-size grid instead
    rng = random.Random(54)
    matrix = [[rng.randint(-100, 100) for _ in range(10)] for _ in range(10)]
    return {"args": [matrix], "expected": _spiral(matrix)}


def _random_case(rng):
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    return {"args": [[[rng.randint(-9, 9) for _ in range(n)] for _ in range(m)]]}


TASK = {
    "title": "Spiral Matrix",
    "title_zh": "螺旋矩阵",
    "difficulty": "Medium",
    "leetcode_id": 54,
    "leetcode_slug": "spiral-matrix",
    "category": "Matrix",
    "function_name": "spiralOrder",
    "params": [("matrix", "int[][]")],
    "returns": "int[]",
    "description_en": "Given an `m x n` matrix, return all of its elements in spiral order: go right along the top row, down the right column, left along the bottom row, up the left column, then repeat the same walk on the inner part.\n\n**Example 1:** `matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]` → `[1,2,3,4,8,12,11,10,9,5,6,7]`\n\n**Example 2:** `matrix = [[1],[2],[3]]` → `[1,2,3]`\n\n**Constraints:**\n- `1 <= m, n <= 10`\n- `-100 <= matrix[i][j] <= 100`",
    "description_zh": "给定 `m x n` 的矩阵，按螺旋顺序返回其中所有元素：先沿最上面一行从左到右，再沿最右边一列从上到下，然后沿最下面一行从右到左，再沿最左边一列从下到上，之后对内部重复同样的路线。\n\n**示例 1：** `matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]` → `[1,2,3,4,8,12,11,10,9,5,6,7]`\n\n**示例 2：** `matrix = [[1],[2],[3]]` → `[1,2,3]`\n\n**约束：**\n- `1 <= m, n <= 10`\n- `-100 <= matrix[i][j] <= 100`",
    "hint": "Peel the matrix one layer at a time with four boundaries (top, bottom, left, right), shrinking them after each lap. Once only one row or column is left, don't walk it twice.",
    "hint_zh": "用上、下、左、右四条边界一层一层地剥开矩阵，每走完一圈就向内收缩。只剩一行或一列时，不要重复访问。",
    "cases": [
        {"args": [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]], "expected": [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]},
        {"args": [[[1], [2], [3]]], "expected": [1, 2, 3]},
        # A single cell, and a single row: the bottom pass must not repeat the one value
        {"args": [[[7]]], "expected": [7]},
        {"args": [[[1, 2, 3]]], "expected": [1, 2, 3]},
        {"args": [[[1, 2], [3, 4]]], "expected": [1, 2, 4, 3]},
        {"args": [[[1, 2, 3], [4, 5, 6]]], "expected": [1, 2, 3, 6, 5, 4]},
        # Tall grid: the left column has to be walked up to the second row
        {"args": [[[-1, 0], [0, -1], [2, 2], [3, 3]]], "expected": [-1, 0, -1, 2, 3, 3, 2, 0]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        out = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        while top <= bottom and left <= right:
            for j in range(left, right + 1):
                out.append(matrix[top][j])
            for i in range(top + 1, bottom + 1):
                out.append(matrix[i][right])
            # The bottom and left passes only exist when a second row and column remain
            if top < bottom and left < right:
                for j in range(right - 1, left, -1):
                    out.append(matrix[bottom][j])
                for i in range(bottom, top, -1):
                    out.append(matrix[i][left])
            top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
        return out''',
    "explanation": "Peel the matrix layer by layer. Four boundaries mark the unvisited region: walk the top row, the right column, the bottom row (right to left) and the left column (bottom to top), then move every boundary inwards by one. The guard `top < bottom and left < right` stops a lone row or column from being walked a second time; the guard is needed for the bottom and left passes only.\n\nTime O(mn), space O(1) beyond the output.",
    "explanation_zh": "逐层剥开矩阵。四条边界划出尚未访问的区域：依次走上边、右边、下边（从右到左）和左边（从下到上），然后让四条边界各向内收缩一格。只有在仍然存在第二行和第二列时才需要走下边和左边，所以用 `top < bottom and left < right` 做判断，避免只剩一行或一列时重复访问。\n\n时间 O(mn)，除输出外空间 O(1)。",
}
