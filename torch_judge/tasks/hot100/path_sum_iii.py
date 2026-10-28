"""Path Sum III (LeetCode 437)."""

import random

from torch_judge.algo.runner import TreeNode, tree_to_array


def _random_shape(rng, n, window):
    """A random binary tree with n nodes, as (left, right) child indices (-1 for none). A new node
    takes a free child slot among the `window` most recently created ones, so a small window
    makes deep, lopsided trees. A child always has a larger index than its parent."""
    left, right = [-1] * n, [-1] * n
    slots = [(0, 0), (0, 1)]  # (node, side): side 0 is the left child, 1 the right child
    for new in range(1, n):
        k = len(slots) - 1 - rng.randrange(min(window, len(slots)))
        slots[k], slots[-1] = slots[-1], slots[k]
        parent, side = slots.pop()
        (left if side == 0 else right)[parent] = new
        slots += [(new, 0), (new, 1)]
    return left, right


def _build(vals, left, right):
    nodes = [TreeNode(v) for v in vals]
    for node, lc, rc in zip(nodes, left, right):
        if lc >= 0:
            node.left = nodes[lc]
        if rc >= 0:
            node.right = nodes[rc]
    return nodes[0]


def _large_case():
    # 1000 nodes with small values, so many downward paths hit the target. The naive method
    # (walk down from every node) is O(n^2), but at n <= 1000 it still finishes in about 50 ms,
    # so this case checks the count rather than speed. The tree is about 800 levels deep
    rng = random.Random(437)
    n = 1000
    vals = [rng.randint(-10, 10) for _ in range(n)]
    left, right = _random_shape(rng, n, window=3)
    target = 4
    count = 0
    for start in range(n):
        stack = [(start, 0)]  # (node, sum of the path from start down to node, exclusive)
        while stack:
            i, acc = stack.pop()
            acc += vals[i]
            if acc == target:
                count += 1
            for c in (left[i], right[i]):
                if c >= 0:
                    stack.append((c, acc))
    return {"args": [tree_to_array(_build(vals, left, right)), target], "expected": count}


def _random_case(rng):
    # Bushy trees with values -2..2: a prefix sum from one branch often matches a path in another
    n = rng.randint(0, 12)
    target = rng.randint(-1, 3)
    if n == 0:
        return {"args": [[], target]}
    left, right = _random_shape(rng, n, window=rng.randint(1, 8))
    vals = [rng.randint(-2, 2) for _ in range(n)]
    return {"args": [tree_to_array(_build(vals, left, right)), target]}


TASK = {
    "title": "Path Sum III",
    "title_zh": "路径总和 III",
    "difficulty": "Medium",
    "leetcode_id": 437,
    "leetcode_slug": "path-sum-iii",
    "category": "Binary Tree",
    "function_name": "pathSum",
    "params": [("root", "TreeNode"), ("targetSum", "int")],
    "returns": "int",
    "description_en": "Given the root of a binary tree and an integer `targetSum`, count the **downward paths** whose node values add up to `targetSum`. A downward path starts at some node and moves only from a parent to one of its children. It may start and end anywhere, so it does not have to touch the root.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [4,1,6,-2,3,null,5]`, `targetSum = 3` → `2` (the single node `3`, and the path `4 → 1 → -2`)\n\n**Example 2:** `root = [1,-1]`, `targetSum = 0` → `1` (the path `1 → -1`)\n\n**Constraints:**\n- The tree has `0` to `1000` nodes\n- $-10^{9} \\le \\texttt{Node.val} \\le 10^{9}$\n- $-1000 \\le \\texttt{targetSum} \\le 1000$",
    "description_zh": "给定二叉树的根节点 `root` 和整数 `targetSum`，统计**向下路径**的数量，要求路径上节点值之和等于 `targetSum`。向下路径从某个节点开始，只能从父节点走到子节点。路径的起点和终点可以是任意节点，因此不一定经过根节点。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [4,1,6,-2,3,null,5]`，`targetSum = 3` → `2`（单独的节点 `3`，以及路径 `4 → 1 → -2`）\n\n**示例 2：** `root = [1,-1]`，`targetSum = 0` → `1`（路径 `1 → -1`）\n\n**约束：**\n- 节点数在 `0` 到 `1000` 之间\n- $-10^{9} \\le \\texttt{Node.val} \\le 10^{9}$\n- $-1000 \\le \\texttt{targetSum} \\le 1000$",
    "hint": "Prefix sums. Keep a count of the prefix sums on the current root path; at each node, the number of earlier prefixes equal to (current prefix - targetSum) is the number of paths ending here. Remove a node's prefix when you leave it.",
    "hint_zh": "前缀和。记录当前根路径上各个前缀和出现的次数；到达一个节点时，之前出现过「当前前缀和 - targetSum」的次数，就是以它结尾的路径数。离开节点时，要把它的前缀和从计数中去掉。",
    "cases": [
        {"args": [[4, 1, 6, -2, 3, None, 5], 3], "expected": 2},
        {"args": [[1, -1], 0], "expected": 1},
        {"args": [[], 5], "expected": 0},
        {"args": [[5], 5], "expected": 1},
        # Every node is 0, so each single node and each parent-child pair counts
        {"args": [[0, 0, 0], 0], "expected": 5},
        {"args": [[6, 2, None, None, 7], 9], "expected": 1},
        # Negative values, and a path that starts below the root
        {"args": [[-2, None, -3], -5], "expected": 1},
        # The prefix sum 5 left behind by the left subtree must not count for the right one
        {"args": [[0, 5, 7], 2], "expected": 0},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        # counts[s]: how many prefix sums equal to s lie on the current root path
        counts = {0: 1}
        total = 0
        # Stack entries are (node, prefix sum above it, entering?). Leaving a node removes its
        # prefix sum from the counts; the walk is iterative, so deep trees are fine
        stack = [(root, 0, True)]
        while stack:
            node, prefix, entering = stack.pop()
            if not entering:
                counts[prefix] -= 1
                continue
            if node is None:
                continue
            prefix += node.val
            total += counts.get(prefix - targetSum, 0)
            counts[prefix] = counts.get(prefix, 0) + 1
            stack.append((node, prefix, False))
            stack.append((node.right, prefix, True))
            stack.append((node.left, prefix, True))
        return total''',
    "explanation": "Prefix sums. Let `P(v)` be the sum of the values from the root down to `v`. The sum of the downward path from an ancestor-or-self `u` down to `v` is `P(v) - P(parent of u)`. So while walking the tree, count how many prefix sums on the current root path equal `P(v) - targetSum`; a hash map holds those counts. When the walk leaves a node, its prefix sum is removed again, so a sibling subtree never sees it. Enter and leave markers on an explicit stack keep the walk iterative.\n\nTime O(n), space O(n).",
    "explanation_zh": "前缀和。设 `P(v)` 为从根到 `v` 的所有节点值之和。从祖先（或自身）`u` 向下到 `v` 的路径和为 `P(v) - P(u 的父节点)`。因此在遍历时，只要统计当前根路径上有多少个前缀和等于 `P(v) - targetSum`，哈希表记录的就是这些计数。离开一个节点时，把它的前缀和从计数中移除，这样兄弟子树就看不到它。用显式栈上的「进入」「离开」标记实现遍历，避免了递归。\n\n时间 O(n)，空间 O(n)。",
}
