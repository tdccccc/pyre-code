"""Maximum Depth of Binary Tree (LeetCode 104)."""

import random

from torch_judge.algo.runner import TreeNode, tree_to_array


def _random_tree(rng, n, lo, hi):
    # Hang each new node on a random free child slot: random shapes, mostly shallow
    if n == 0:
        return None
    root = TreeNode(rng.randint(lo, hi))
    slots = [(root, "left"), (root, "right")]
    for _ in range(n - 1):
        parent, side = slots.pop(rng.randrange(len(slots)))
        child = TreeNode(rng.randint(lo, hi))
        setattr(parent, side, child)
        slots += [(child, "left"), (child, "right")]
    return root


def _depth(root):
    # Count the levels: each round replaces the current level with its children
    depth, level = 0, [root] if root is not None else []
    while level:
        depth += 1
        level = [c for node in level for c in (node.left, node.right) if c is not None]
    return depth


def _large_case():
    # 10^4 nodes. The intended O(n) cannot time out at this size, so this checks scale only
    rng = random.Random(104)
    root = _random_tree(rng, 10000, -100, 100)
    return {"args": [tree_to_array(root)], "expected": _depth(root)}


def _random_case(rng):
    return {"args": [tree_to_array(_random_tree(rng, rng.randint(0, 9), -10, 10))]}


TASK = {
    "title": "Maximum Depth of Binary Tree",
    "title_zh": "二叉树的最大深度",
    "difficulty": "Easy",
    "leetcode_id": 104,
    "leetcode_slug": "maximum-depth-of-binary-tree",
    "category": "Binary Tree",
    "function_name": "maxDepth",
    "params": [("root", "TreeNode")],
    "returns": "int",
    "description_en": "Given the `root` of a binary tree, return its depth: the number of nodes on the longest path that starts at the root and ends at a leaf.\n\nTrees are written in level order, with `null` for a missing child. An empty tree has depth `0`.\n\n**Example 1:** `root = [2,7,null,null,6]` → `3` (the path 2 → 7 → 6)\n\n**Example 2:** `root = [0,-3,1,null,-8]` → `3` (the path 0 → -3 → -8)\n\n**Constraints:**\n- The tree has `0` to $10^{4}$ nodes\n- $-100 \\le \\texttt{Node.val} \\le 100$",
    "description_zh": "给定二叉树的根节点 `root`，返回它的深度：从根节点出发、到某个叶子节点结束的所有路径中，节点数最多的那条路径所包含的节点数。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。空树的深度为 `0`。\n\n**示例 1：** `root = [2,7,null,null,6]` → `3`（路径 2 → 7 → 6）\n\n**示例 2：** `root = [0,-3,1,null,-8]` → `3`（路径 0 → -3 → -8）\n\n**约束：**\n- 节点数在 `0` 到 $10^{4}$ 之间\n- $-100 \\le \\texttt{Node.val} \\le 100$",
    "hint": "The depth of a tree is 1 plus the larger depth of its two subtrees, and the depth of an empty tree is 0. Let each node ask its children first.",
    "hint_zh": "一棵树的深度等于两个子树中较大的深度再加 1，空树的深度为 0。让每个节点先问它的子节点。",
    "cases": [
        {"args": [[2, 7, None, None, 6]], "expected": 3},
        {"args": [[0, -3, 1, None, -8]], "expected": 3},
        {"args": [[]], "expected": 0},
        {"args": [[1]], "expected": 1},
        # A node with only a left child: a version that takes the minimum over children says 1
        {"args": [[1, 2]], "expected": 2},
        {"args": [[1, None, 2, None, 3, None, 4]], "expected": 4},
        # The deepest leaf is under the right side, reached through 12 -> 9 -> 4
        {"args": [[7, 3, 12, 1, None, 9, None, None, None, 4]], "expected": 4},
        {"args": [[1, 2, None, 3, None, 4]], "expected": 4},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))''',
    "explanation": "Divide and conquer. An empty tree has depth 0. For any other node, the deepest path either goes into the left subtree or into the right one, so the depth is 1 plus the larger of the two subtree depths.\n\nTime O(n), space O(h) for the recursion, where h is the height of the tree.",
    "explanation_zh": "分治。空树深度为 0。对于其他节点，最深的路径要么走进左子树，要么走进右子树，所以深度等于 1 加上两个子树深度中较大的那个。\n\n时间 O(n)，递归空间 O(h)，其中 h 为树的高度。",
}
