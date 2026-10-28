"""Diameter of Binary Tree (LeetCode 543)."""

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


def _diameter(root):
    # Heights (in nodes) bottom-up over a BFS order, without recursion. A node's best path
    # joins its two children's chains; the answer counts edges, which is the same as nodes
    order = [root]
    for node in order:
        order += [c for c in (node.left, node.right) if c is not None]
    height = {}
    best = 0
    for node in reversed(order):
        left = height[id(node.left)] if node.left is not None else 0
        right = height[id(node.right)] if node.right is not None else 0
        height[id(node)] = 1 + max(left, right)
        best = max(best, left + right)
    return best


def _balanced(values):
    # Midpoint recursion: a balanced tree (depth about log2 n) holding the values in order
    if not values:
        return None
    mid = len(values) // 2
    return TreeNode(values[mid], _balanced(values[:mid]), _balanced(values[mid + 1:]))


def _large_case():
    # A 9000-node chain with random left/right turns, ending in a balanced 1000-node tree, so
    # the depth is about 9000. The naive method (recomputing heights at every node) costs the
    # sum of all node depths, about 5 * 10^7 recursive calls: several seconds. The depth stays
    # within the judge's recursion limit (20000 in both modes), so a recursive O(n) answer passes
    rng = random.Random(543)
    root = TreeNode(rng.randint(-100, 100))
    node = root
    for _ in range(8999):
        child = TreeNode(rng.randint(-100, 100))
        if rng.random() < 0.5:
            node.left = child
        else:
            node.right = child
        node = child
    node.left = _balanced(sorted(rng.randint(-100, 100) for _ in range(1000)))
    return {"args": [tree_to_array(root)], "expected": _diameter(root)}


def _random_case(rng):
    return {"args": [tree_to_array(_random_tree(rng, rng.randint(4, 14), -9, 9))]}


TASK = {
    "title": "Diameter of Binary Tree",
    "title_zh": "二叉树的直径",
    "difficulty": "Easy",
    "leetcode_id": 543,
    "leetcode_slug": "diameter-of-binary-tree",
    "category": "Binary Tree",
    "function_name": "diameterOfBinaryTree",
    "params": [("root", "TreeNode")],
    "returns": "int",
    "description_en": "Given the `root` of a binary tree, return its **diameter**: the number of edges on the longest route between any two nodes. That route may pass through the root or stay inside one subtree.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [1,2,null,3,6,4,null,null,7,5]` → `5` (the route 5 → 4 → 3 → 2 → 6 → 7 never touches the root)\n\n**Example 2:** `root = [4,-1]` → `1`\n\n**Constraints:**\n- The tree has `1` to `10^4` nodes\n- `-100 <= Node.val <= 100`",
    "description_zh": "给定二叉树的根节点 `root`，返回它的**直径**：树中任意两个节点之间最长路径所含的边数。这条路径可以经过根节点，也可以只在某一棵子树内部。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [1,2,null,3,6,4,null,null,7,5]` → `5`（路径 5 → 4 → 3 → 2 → 6 → 7 没有经过根节点）\n\n**示例 2：** `root = [4,-1]` → `1`\n\n**约束：**\n- 节点数在 `1` 到 `10^4` 之间\n- `-100 <= Node.val <= 100`",
    "hint": "For each node, the longest route that bends at it is the height of its left subtree plus the height of its right subtree. Compute heights bottom-up and record the best bend along the way.",
    "hint_zh": "对每个节点，以它为转折点的最长路径等于左子树高度加上右子树高度。自底向上计算高度，同时记录沿途的最佳转折。",
    "cases": [
        {"args": [[1, 2, None, 3, 6, 4, None, None, 7, 5]], "expected": 5},
        {"args": [[4, -1]], "expected": 1},
        {"args": [[1]], "expected": 0},
        {"args": [[2, 4, None, 6, None, None, 8]], "expected": 3},
        # Complete tree of 7: the route 4 -> 2 -> 1 -> 3 -> 6 goes through the root
        {"args": [[1, 2, 3, 4, 5, 6, 7]], "expected": 4},
        {"args": [[1, None, 2, None, 3, None, 4]], "expected": 3},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        # Breadth-first order lists every parent before its children, so reversed it lists
        # children first and each node's child heights are ready when the node is reached
        order = [root]
        for node in order:
            order += [c for c in (node.left, node.right) if c is not None]
        height = {None: 0}
        best = 0
        for node in reversed(order):
            left, right = height[node.left], height[node.right]
            height[node] = 1 + max(left, right)
            best = max(best, left + right)
        return best''',
    "explanation": "One pass that returns heights, done bottom-up without recursion. A node's height is 1 plus the taller child height. Every node also yields a candidate route that bends at it, of length left height plus right height, and the diameter is the largest candidate. Each node is handled once, so the route through the root is not special.\n\nTime O(n), space O(n) for the order and the heights.",
    "explanation_zh": "一次自底向上的遍历同时计算高度，不使用递归。节点的高度等于较高子节点的高度加 1。每个节点还能得到一条以它为转折点的候选路径，长度为左子树高度加右子树高度，直径就是所有候选中的最大值。每个节点只处理一次，所以经过根节点的路径并不特殊。\n\n时间 O(n)，空间 O(n)（用于保存遍历顺序和高度）。",
}
