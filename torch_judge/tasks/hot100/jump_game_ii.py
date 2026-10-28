"""Jump Game II (LeetCode 45)."""

import random
from collections import deque


def _min_jumps(nums: list[int]) -> int:
    # Breadth-first search by jump count; -1 if the last index is unreachable
    dist = [-1] * len(nums)
    dist[0] = 0
    queue = deque([0])
    while queue:
        i = queue.popleft()
        for j in range(i + 1, min(len(nums), i + nums[i] + 1)):
            if dist[j] == -1:
                dist[j] = dist[i] + 1
                queue.append(j)
    return dist[-1]


def _large_case():
    # 10^4 indices, each with a jump of 1 or 2: about 5000 jumps are needed, and there are
    # Fibonacci-many routes, so a search that tries every route never finishes. (An O(n²) DP
    # takes about 1 s here, so it is not caught by time.)
    rng = random.Random(45)
    nums = [rng.randint(1, 2) for _ in range(10000)]
    return {"args": [nums], "expected": _min_jumps(nums)}


def _random_case(rng):
    # Retry until the last index is reachable, which the problem guarantees
    while True:
        nums = [rng.randint(0, 3) for _ in range(rng.randint(1, 8))]
        if _min_jumps(nums) != -1:
            return {"args": [nums]}


TASK = {
    "title": "Jump Game II",
    "title_zh": "跳跃游戏 II",
    "difficulty": "Medium",
    "leetcode_id": 45,
    "leetcode_slug": "jump-game-ii",
    "category": "Greedy",
    "function_name": "jump",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "You start on index `0` of the integer array `nums`. From index `i` you may jump forward to any index from $\\texttt{i} + 1$ up to $\\texttt{i} + \\texttt{nums[i]}$. The last index is always reachable.\n\nReturn the fewest jumps needed to get from index `0` to the last index.\n\n**Example 1:** `nums = [4,1,1,3,1]` → `1` (one jump from index 0 straight to the end)\n\n**Example 2:** `nums = [1,3,2,1,1]` → `2` (index 0 → 1 → 4)\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 10^{4}$\n- $0 \\le \\texttt{nums[i]} \\le 1000$\n- The last index is reachable from index 0",
    "description_zh": "你从整数数组 `nums` 的下标 `0` 出发。在下标 `i` 处，可以向前跳到 $\\texttt{i} + 1$ 到 $\\texttt{i} + \\texttt{nums[i]}$ 之间的任意下标。保证最后一个下标一定可达。\n\n返回到达最后一个下标所需的最少跳跃次数。\n\n**示例 1：** `nums = [4,1,1,3,1]` → `1`（从下标 0 一步跳到末尾）\n\n**示例 2：** `nums = [1,3,2,1,1]` → `2`（下标 0 → 1 → 4）\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 10^{4}$\n- $0 \\le \\texttt{nums[i]} \\le 1000$\n- 从下标 0 出发一定能到达最后一个下标",
    "hint": "Think in layers. Everything reachable with k jumps forms a contiguous block of indices. Jump only when you run out of the current block, and choose the farthest landing point.",
    "hint_zh": "按层思考：用 k 次跳跃能到达的下标构成一段连续区间。只有当当前区间走完时才多跳一次，并且跳到区间内能到达的最远位置。",
    "cases": [
        {"args": [[4, 1, 1, 3, 1]], "expected": 1},
        {"args": [[1, 3, 2, 1, 1]], "expected": 2},
        {"args": [[0]], "expected": 0},
        # Moving the window end before folding in index i's reach undercounts this one as 1
        {"args": [[1, 1, 1]], "expected": 2},
        {"args": [[1, 1, 1, 1]], "expected": 3},
        {"args": [[2, 0, 2, 1, 1]], "expected": 2},
        {"args": [[1, 2, 1, 1, 1]], "expected": 3},
        {"args": [[2, 1, 2, 1, 1]], "expected": 2},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = end = farthest = 0  # end: last index reachable with `jumps` jumps
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])  # reach if one more jump is taken from here
            if i == end:  # the current layer is used up, so one more jump is needed
                jumps += 1
                end = farthest
        return jumps''',
    "explanation": "Process indices in layers. `end` is the last index reachable with the jumps counted so far, and `farthest` is the furthest index reachable with one more jump. Each index is folded into `farthest` before the scan checks whether it reached `end`. When the scan reaches `end`, the current layer is used up, so one more jump is counted and `end` moves to `farthest`. Taking the farthest landing point in each layer is never worse than any other choice.\n\nTime O(n), space O(1).",
    "explanation_zh": "按层处理下标。`end` 表示用已经计入的跳数能到达的最后一个下标，`farthest` 表示再跳一次能到达的最远下标。扫描每个下标时，先用它更新 `farthest`，再检查是否走到了 `end`。走到 `end` 说明当前这一层用完，需要多跳一次，并把 `end` 推进到 `farthest`。每一层都选择能到达的最远位置，不会比其他选择差。\n\n时间 O(n)，空间 O(1)。",
}
