"""Binary Tree Maximum Path Sum (LeetCode 124)."""

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
    # 3 * 10^4 nodes in a deep tree (about 7700 levels): a recursive DFS needs thousands of
    # frames, within the judge's recursion limit. The expected value comes from a pass over
    # the index arrays
    rng = random.Random(124)
    n = 30000
    vals = [rng.randint(-1000, 1000) for _ in range(n)]
    left, right = _random_shape(rng, n, window=10)
    gain = [0] * n  # best downward path sum that starts at each node
    best = None
    for i in reversed(range(n)):  # a child always has a larger index than its parent
        lg = max(gain[left[i]], 0) if left[i] >= 0 else 0
        rg = max(gain[right[i]], 0) if right[i] >= 0 else 0
        through = vals[i] + lg + rg
        if best is None or through > best:
            best = through
        gain[i] = vals[i] + max(lg, rg)
    return {"args": [tree_to_array(_build(vals, left, right))], "expected": best}


def _random_case(rng):
    n = rng.randint(1, 7)
    left, right = _random_shape(rng, n, window=rng.randint(1, 3))
    vals = [rng.randint(-10, 10) for _ in range(n)]
    return {"args": [tree_to_array(_build(vals, left, right))]}


TASK = {
    "title": "Binary Tree Maximum Path Sum",
    "title_zh": "二叉树中的最大路径和",
    "difficulty": "Hard",
    "leetcode_id": 124,
    "leetcode_slug": "binary-tree-maximum-path-sum",
    "category": "Binary Tree",
    "function_name": "maxPathSum",
    "params": [("root", "TreeNode")],
    "returns": "int",
    "description_en": "A **path** in a binary tree is a sequence of nodes in which each pair of neighbouring nodes is joined by an edge. Each node appears at most once. A path does not have to pass through the root, and it must contain at least one node.\n\nReturn the largest possible sum of the values of the nodes on such a path.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [1,2,3]` → `6` (the path `2 → 1 → 3`)\n\n**Example 2:** `root = [-3,6,2,1,null,null,5]` → `11` (the path `1 → 6 → -3 → 2 → 5` covers the whole tree)\n\n**Constraints:**\n- The tree has `1` to `3 * 10^4` nodes\n- `-1000 <= Node.val <= 1000`",
    "description_zh": "二叉树中的**路径**是一串节点，其中每两个相邻节点之间都有一条边相连。每个节点在路径中最多出现一次。路径不一定经过根节点，并且至少包含一个节点。\n\n返回这样一条路径上所有节点值之和的最大可能值。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [1,2,3]` → `6`（路径 `2 → 1 → 3`）\n\n**示例 2：** `root = [-3,6,2,1,null,null,5]` → `11`（路径 `1 → 6 → -3 → 2 → 5` 恰好覆盖整棵树）\n\n**约束：**\n- 节点数在 `1` 到 `3 * 10^4` 之间\n- `-1000 <= Node.val <= 1000`",
    "hint": "For each node, compute the best downward path that starts at it and goes into at most one child (or stops there). The best path that bends at this node is its value plus the two best child gains. Negative gains should be dropped. Process children before their parents.",
    "hint_zh": "对每个节点，计算从它出发、最多延伸到一个子节点的最大向下路径和（也可以只包含它自己）。以该节点为拐点的最优路径，是它的值加上两侧子节点的最大贡献。负的贡献应当舍弃。注意要先处理子节点，再处理父节点。",
    "cases": [
        {"args": [[1, 2, 3]], "expected": 6},
        {"args": [[-3, 6, 2, 1, None, None, 5]], "expected": 11},
        # A single negative node: the answer is that node, not 0
        {"args": [[-3]], "expected": -3},
        # The best path is the root alone: a negative child must not be added
        {"args": [[2, -1]], "expected": 2},
        # Every value is negative: the best path is the least negative single node
        {"args": [[-2, -1]], "expected": -1},
        # The best path bends at 2, below the root, so it never goes through the root
        {"args": [[1, 2, -4, 7, 8, None, 9]], "expected": 17},
        {"args": [[-100, 3, 5, 1, 2]], "expected": 6},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # Collect the nodes in a walk from the root, so that reversing the list puts every
        # child before its parent. The walk is iterative, so deep trees are fine
        order, stack = [], [root]
        while stack:
            node = stack.pop()
            order.append(node)
            for child in (node.left, node.right):
                if child:
                    stack.append(child)
        gain = {}  # node -> best downward path sum that starts at node
        best = float("-inf")
        for node in reversed(order):
            left = max(gain[node.left], 0) if node.left else 0
            right = max(gain[node.right], 0) if node.right else 0
            # The path that bends at this node uses both branches; negative branches are dropped
            best = max(best, node.val + left + right)
            gain[node] = node.val + max(left, right)
        return best''',
    "explanation": "Post-order DP. For each node, `gain` is the best path sum that starts at the node and goes down into at most one child. Such a path can be extended upward by the parent, so it is the value plus the larger non-negative child gain. Any path has a highest node, and at that node the path is the value plus up to two non-negative child gains, which is what is recorded as the candidate answer. Children are processed before parents by reversing a walk from the root, so no recursion is needed.\n\nTime O(n), space O(n).",
    "explanation_zh": "后序动态规划。对每个节点，`gain` 表示从该节点出发、向下最多延伸到一个子节点的最大路径和。这样的路径可以再向上接到父节点，所以它等于节点值加上较大的非负子节点贡献。任意一条路径都有一个最高点，在这个点上路径等于节点值加上至多两个非负的子节点贡献，这就是候选答案。通过把从根出发的遍历结果逆序，子节点总是先于父节点处理，因此不需要递归。\n\n时间 O(n)，空间 O(n)。",
}
