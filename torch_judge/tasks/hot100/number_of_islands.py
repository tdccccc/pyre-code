"""Number of Islands (LeetCode 200)."""

import random


def _large_case():
    # 300 x 300 with about 60% land, just above the site percolation threshold, so one giant
    # island spans much of the grid. A recursive flood fill would recurse far past Python's limit
    rng = random.Random(200)
    m = n = 300
    grid = [["1" if rng.random() < 0.6 else "0" for _ in range(n)] for _ in range(m)]
    # Union-find over the land cells: a different method from the reference's BFS
    parent = list(range(m * n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(n):
            if grid[i][j] != "1":
                continue
            for x, y in ((i + 1, j), (i, j + 1)):
                if x < m and y < n and grid[x][y] == "1":
                    parent[find(i * n + j)] = find(x * n + y)
    islands = sum(1 for i in range(m) for j in range(n)
                  if grid[i][j] == "1" and find(i * n + j) == i * n + j)
    return {"args": [grid], "expected": islands}


def _random_case(rng):
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    return {"args": [[[rng.choice("01") for _ in range(n)] for _ in range(m)]]}


TASK = {
    "title": "Number of Islands",
    "title_zh": "岛屿数量",
    "difficulty": "Medium",
    "leetcode_id": 200,
    "leetcode_slug": "number-of-islands",
    "category": "Graphs",
    "function_name": "numIslands",
    "params": [("grid", "char[][]")],
    "returns": "int",
    "description_en": "Given an `m x n` grid whose cells are `\"1\"` (land) or `\"0\"` (water), return the number of islands. An island is a group of land cells connected horizontally or vertically; cells touching only diagonally are not connected. Cells outside the grid count as water.\n\n**Example 1:** `grid = [[\"1\",\"1\",\"0\"],[\"0\",\"1\",\"0\"],[\"1\",\"0\",\"1\"]]` → `3`\n\n**Example 2:** `grid = [[\"0\",\"1\",\"0\",\"1\"],[\"1\",\"0\",\"1\",\"0\"]]` → `4` (the land cells touch only diagonally, so every one is its own island)\n\n**Constraints:**\n- `1 <= m, n <= 300`\n- Every cell is `\"1\"` or `\"0\"`",
    "description_zh": "给定一个 `m x n` 的网格，每个格子是 `\"1\"`（陆地）或 `\"0\"`（水）。返回岛屿的数量。岛屿是由水平或竖直方向相连的陆地格子组成的区域；只有对角线相接的陆地不算相连。网格之外视为水。\n\n**示例 1：** `grid = [[\"1\",\"1\",\"0\"],[\"0\",\"1\",\"0\"],[\"1\",\"0\",\"1\"]]` → `3`\n\n**示例 2：** `grid = [[\"0\",\"1\",\"0\",\"1\"],[\"1\",\"0\",\"1\",\"0\"]]` → `4`（陆地之间只有对角相接，因此每一块都是独立的岛屿）\n\n**约束：**\n- `1 <= m, n <= 300`\n- 每个格子都是 `\"1\"` 或 `\"0\"`",
    "hint": "Scan the grid. Each time you meet an unvisited land cell, you have found a new island: flood-fill everything connected to it, using a queue or stack, so that the whole island is marked and not counted again.",
    "hint_zh": "逐个扫描网格。每遇到一个未访问过的陆地格子，就找到了一个新岛屿：用队列或栈把与它相连的所有陆地都标记掉，这样整座岛只会计数一次。",
    "cases": [
        {"args": [[["1", "1", "0"], ["0", "1", "0"], ["1", "0", "1"]]], "expected": 3},
        {"args": [[["0", "1", "0", "1"], ["1", "0", "1", "0"]]], "expected": 4},
        {"args": [[["1"]]], "expected": 1},
        {"args": [[["0"]]], "expected": 0},
        {"args": [[["1", "1", "1"], ["1", "1", "1"]]], "expected": 1},
        # The upper-right cell is found first. A flood fill that only moves down and right
        # never reaches the lower-left arm of this U shape and counts it again
        {"args": [[["0", "0", "1"], ["1", "1", "1"]]], "expected": 1},
        {"args": [[["1", "0", "1", "0", "1"]]], "expected": 3},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import deque


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] != "1":
                    continue
                islands += 1
                # Mark the island as it is found, so each island is counted exactly once
                grid[i][j] = "0"
                queue = deque([(i, j)])
                while queue:
                    r, c = queue.popleft()
                    for x, y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if 0 <= x < m and 0 <= y < n and grid[x][y] == "1":
                            grid[x][y] = "0"
                            queue.append((x, y))
        return islands''',
    "explanation": "Scan the cells in order. Every land cell that is still unmarked when the scan reaches it starts a new island, so the counter goes up by one, and a breadth-first flood fill then marks every land cell connected to it. Marking cells as they are queued keeps each cell in the queue at most once. The queue is used instead of recursion, because one island can contain up to 90,000 cells.\n\nTime O(m * n), space O(m * n).",
    "explanation_zh": "按顺序扫描每个格子。扫描到一个尚未标记的陆地格子时，就出现了一座新岛屿，计数加一；然后用广度优先的洪水填充把与它相连的陆地全部标记。格子在入队时就标记，所以每个格子只会入队一次。这里用队列而不用递归，因为一座岛可能包含约 9 万个格子。\n\n时间 O(m * n)，空间 O(m * n)。",
}
