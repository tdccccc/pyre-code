"""Course Schedule (LeetCode 207)."""

import random


def _large_case():
    # 2000 courses and 5000 prerequisites, all pointing from a course to a later-numbered one,
    # so the answer is true. A chain of 2000 courses is included: a recursive DFS needs 2000
    # frames there, past Python's default limit in ACM programs. (Rescanning until no course
    # is free takes ~2000 passes, but at these sizes it still runs in about half a second.)
    rng = random.Random(207)
    n = 2000
    edges = {(i, i + 1) for i in range(n - 1)}
    while len(edges) < 5000:
        edges.add(tuple(sorted(rng.sample(range(n), 2))))
    pairs = [list(e) for e in sorted(edges)]
    rng.shuffle(pairs)
    # Every prerequisite is a later-numbered course, so taking courses from n-1 down to 0 works
    expected = all(a < b for a, b in pairs)
    return {"args": [n, pairs], "expected": expected}


def _random_case(rng):
    n = rng.randint(1, 6)
    pairs = [[a, b] for a in range(n) for b in range(n) if a != b]
    chosen = rng.sample(pairs, rng.randint(0, min(len(pairs), 7)))
    return {"args": [n, chosen]}


TASK = {
    "title": "Course Schedule",
    "title_zh": "课程表",
    "difficulty": "Medium",
    "leetcode_id": 207,
    "leetcode_slug": "course-schedule",
    "category": "Graphs",
    "function_name": "canFinish",
    "params": [("numCourses", "int"), ("prerequisites", "int[][]")],
    "returns": "bool",
    "description_en": "There are `numCourses` courses, labelled `0` to $\\texttt{numCourses} - 1$. For `prerequisites[i] = [a, b]`, course `a` can only be taken after course `b` has been finished.\n\nReturn `true` if one order of study lets you finish every course, or `false` if the prerequisites contradict each other.\n\n**Example 1:** `numCourses = 5`, `prerequisites = [[1,0],[2,0],[3,1],[3,2],[4,3]]` → `true` (study `0`, then `1` and `2`, then `3`, then `4`)\n\n**Example 2:** `numCourses = 3`, `prerequisites = [[0,2],[1,0],[2,1]]` → `false` (course `0` needs `2`, which needs `1`, which needs `0`)\n\n**Constraints:**\n- $1 \\le \\texttt{numCourses} \\le 2000$\n- $0 \\le \\texttt{prerequisites.length} \\le 5000$\n- `prerequisites[i] = [a, b]` with $0 \\le \\texttt{a},\\ \\texttt{b} < \\texttt{numCourses}$ and $\\texttt{a} \\ne \\texttt{b}$",
    "description_zh": "一共有 `numCourses` 门课程，编号从 `0` 到 $\\texttt{numCourses} - 1$。对于 `prerequisites[i] = [a, b]`，只有先修完课程 `b`，才能修课程 `a`。\n\n如果存在一个学习顺序，能让你修完所有课程，返回 `true`；如果先修要求自相矛盾，返回 `false`。\n\n**示例 1：** `numCourses = 5`，`prerequisites = [[1,0],[2,0],[3,1],[3,2],[4,3]]` → `true`（依次学 `0`，然后 `1` 和 `2`，再学 `3`，最后学 `4`）\n\n**示例 2：** `numCourses = 3`，`prerequisites = [[0,2],[1,0],[2,1]]` → `false`（课程 `0` 需要 `2`，`2` 需要 `1`，`1` 又需要 `0`）\n\n**约束：**\n- $1 \\le \\texttt{numCourses} \\le 2000$\n- $0 \\le \\texttt{prerequisites.length} \\le 5000$\n- `prerequisites[i] = [a, b]`，其中 $0 \\le \\texttt{a},\\ \\texttt{b} < \\texttt{numCourses}$ 且 $\\texttt{a} \\ne \\texttt{b}$",
    "hint": "Kahn's algorithm: count the unfinished prerequisites of each course. Repeatedly take a course whose count is zero and reduce the counts of the courses it unlocks. If you cannot take every course, the leftover ones form a cycle.",
    "hint_zh": "Kahn 算法：统计每门课还未修完的先修课数量。反复选取数量为零的课程，并减少它所解锁的课程的计数。如果不能修完全部课程，剩下的课程必然构成环。",
    "cases": [
        {"args": [5, [[1, 0], [2, 0], [3, 1], [3, 2], [4, 3]]], "expected": True},
        {"args": [3, [[0, 2], [1, 0], [2, 1]]], "expected": False},
        {"args": [1, []], "expected": True},
        {"args": [2, [[0, 1], [1, 0]]], "expected": False},
        {"args": [4, []], "expected": True},
        # A cycle that no course outside it depends on: one contradictory loop is enough
        {"args": [6, [[1, 0], [2, 1], [5, 4], [4, 5]]], "expected": False},
        # A cycle through all four courses, closed by the last prerequisite
        {"args": [4, [[1, 0], [2, 1], [3, 2], [0, 3]]], "expected": False},
        # Course 0 is needed by both 1 and 2. A visited-set search would mistake reaching it twice for a loop
        {"args": [4, [[1, 0], [2, 0], [3, 1], [3, 2]]], "expected": True},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import deque


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # Unfinished prerequisites per course, and the courses each course unlocks
        pending = [0] * numCourses
        unlocks = [[] for _ in range(numCourses)]
        for course, prereq in prerequisites:
            pending[course] += 1
            unlocks[prereq].append(course)
        ready = deque(c for c in range(numCourses) if pending[c] == 0)
        taken = 0
        while ready:
            c = ready.popleft()
            taken += 1
            for nxt in unlocks[c]:
                pending[nxt] -= 1
                if pending[nxt] == 0:
                    ready.append(nxt)
        # A course that never reaches zero pending prerequisites is stuck in or behind a cycle
        return taken == numCourses''',
    "explanation": "Topological sort with Kahn's algorithm. Courses with no unfinished prerequisites can be taken now, and taking one removes it from the pending count of every course it unlocks. If the order can be built, every course gets taken. If there is a cycle, none of its courses ever reaches zero pending prerequisites, so the number taken falls short of `numCourses`.\n\nTime O(numCourses + len(prerequisites)), space O(numCourses + len(prerequisites)).",
    "explanation_zh": "使用 Kahn 算法做拓扑排序。没有未修先修课的课程现在就能修；修完一门课，就把它从所有被它解锁的课程的待修数中减一。如果能排出顺序，每门课都会被修完。如果存在环，环上的课程的待修数永远无法降到零，于是修完的课程数少于 `numCourses`。\n\n时间 O(numCourses + len(prerequisites))，空间 O(numCourses + len(prerequisites))。",
}
