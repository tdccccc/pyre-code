"""Rotting Oranges (LeetCode 994)."""

import random


def _large_case():
    # 10 x 10 is the constraint limit, so this case checks the answer on a random mix rather
    # than speed: any correct method is already fast at this size
    rng = random.Random(994)
    grid = [[rng.choice([0, 1, 1, 1, 2]) for _ in range(10)] for _ in range(10)]
    # Minute-by-minute simulation on a copy: a different method from the reference's BFS
    g = [row[:] for row in grid]
    minutes = 0
    while True:
        spread = {(x, y) for i in range(10) for j in range(10) if g[i][j] == 2
                  for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                  if 0 <= x < 10 and 0 <= y < 10 and g[x][y] == 1}
        if not spread:
            break
        for x, y in spread:
            g[x][y] = 2
        minutes += 1
    expected = -1 if any(1 in row for row in g) else minutes
    return {"args": [grid], "expected": expected}


def _random_case(rng):
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    return {"args": [[[rng.choice([0, 1, 1, 2]) for _ in range(n)] for _ in range(m)]]}


TASK = {
    "title": "Rotting Oranges",
    "title_zh": "腐烂的橘子",
    "difficulty": "Medium",
    "leetcode_id": 994,
    "leetcode_slug": "rotting-oranges",
    "category": "Graphs",
    "function_name": "orangesRotting",
    "params": [("grid", "int[][]")],
    "returns": "int",
    "description_en": "You are given an `m x n` grid. Each cell is an empty cell (`0`), a fresh orange (`1`) or a rotten orange (`2`). Every minute, each fresh orange that is directly next to a rotten one (up, down, left or right) becomes rotten.\n\nReturn the number of minutes until no fresh orange is left, or `-1` if some fresh orange can never be reached. If there are no fresh oranges at the start, the answer is `0`.\n\n**Example 1:** `grid = [[2,1],[0,1]]` → `2` (the top-right orange rots in minute 1, the bottom-right one in minute 2)\n\n**Example 2:** `grid = [[2,0,1]]` → `-1` (the empty cell walls off the last orange)\n\n**Constraints:**\n- `1 <= m, n <= 10`\n- `grid[i][j]` is `0`, `1` or `2`",
    "description_zh": "给定一个 `m x n` 的网格，每个格子是空地（`0`）、新鲜橘子（`1`）或腐烂橘子（`2`）。每分钟，与腐烂橘子上下左右相邻的新鲜橘子都会变成腐烂。\n\n返回所有新鲜橘子都腐烂所需的分钟数；如果有新鲜橘子永远无法被波及，返回 `-1`。如果开始时就没有新鲜橘子，返回 `0`。\n\n**示例 1：** `grid = [[2,1],[0,1]]` → `2`（右上角的橘子在第 1 分钟腐烂，右下角的在第 2 分钟腐烂）\n\n**示例 2：** `grid = [[2,0,1]]` → `-1`（空地把最后一个橘子隔开了）\n\n**约束：**\n- `1 <= m, n <= 10`\n- `grid[i][j]` 为 `0`、`1` 或 `2`",
    "hint": "Multi-source BFS. Put every rotten orange in the queue at the start, then process the queue one minute at a time: each round rots the fresh neighbours of the current frontier. Count the rounds that rot something.",
    "hint_zh": "多源广度优先搜索。开始时把所有腐烂的橘子都放进队列，然后一分钟一轮地处理：每轮让当前边缘上的新鲜邻居腐烂，并统计真正发生腐烂的轮数。",
    "cases": [
        {"args": [[[2, 1], [0, 1]]], "expected": 2},
        {"args": [[[2, 0, 1]]], "expected": -1},
        {"args": [[[0]]], "expected": 0},
        {"args": [[[1]]], "expected": -1},
        {"args": [[[2]]], "expected": 0},
        {"args": [[[2, 1, 1], [1, 1, 1], [0, 1, 2]]], "expected": 2},
        # One orange rots per minute along the row. Spreading within a single sweep of the
        # grid would finish in one minute
        {"args": [[[2, 1, 1, 1, 1]]], "expected": 4},
        {"args": [[[2, 1, 0, 1]]], "expected": -1},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import deque


class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j, 0))
                elif grid[i][j] == 1:
                    fresh += 1
        minutes = 0
        while queue:
            r, c, t = queue.popleft()
            minutes = max(minutes, t)
            for x, y in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                    grid[x][y] = 2
                    fresh -= 1
                    queue.append((x, y, t + 1))
        return minutes if fresh == 0 else -1''',
    "explanation": "Multi-source BFS. All rotten oranges start in the queue at minute 0. Each orange is queued with the minute it rots, so a fresh neighbour of an orange rotten at minute `t` rots at minute `t + 1`. The answer is the largest such minute, provided every fresh orange was reached; otherwise some orange is walled off and the answer is -1. A grid with no fresh oranges gives 0 because the maximum starts at 0.\n\nTime O(m * n), space O(m * n).",
    "explanation_zh": "多源广度优先搜索。开始时所有腐烂的橘子在第 0 分钟入队。每个橘子入队时带上它腐烂的分钟数，于是在第 `t` 分钟腐烂的橘子的新鲜邻居会在第 `t + 1` 分钟腐烂。答案就是这些分钟数中的最大值，前提是所有新鲜橘子都被波及；否则说明有橘子被隔开，答案为 -1。如果一开始就没有新鲜橘子，最大值从 0 开始，结果也是 0。\n\n时间 O(m * n)，空间 O(m * n)。",
}
