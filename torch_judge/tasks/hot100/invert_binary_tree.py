"""Invert Binary Tree (LeetCode 226)."""

import random

from torch_judge.algo.runner import TreeNode, build_tree, tree_to_array


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


def _mirror(root):
    # Swap children at every node with an explicit stack; returns the same root
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        node.left, node.right = node.right, node.left
        for child in (node.left, node.right):
            if child is not None:
                stack.append(child)
    return root


def _large_case():
    # 10^4 nodes. The intended O(n) cannot time out at this size, so this checks scale only
    rng = random.Random(226)
    root = _random_tree(rng, 10000, -100, 100)
    args = tree_to_array(root)
    return {"args": [args], "expected": tree_to_array(_mirror(build_tree(args)))}


def _random_case(rng):
    return {"args": [tree_to_array(_random_tree(rng, rng.randint(0, 9), -9, 9))]}


TASK = {
    "title": "Invert Binary Tree",
    "title_zh": "翻转二叉树",
    "difficulty": "Easy",
    "leetcode_id": 226,
    "leetcode_slug": "invert-binary-tree",
    "category": "Binary Tree",
    "function_name": "invertTree",
    "params": [("root", "TreeNode")],
    "returns": "TreeNode",
    "description_en": "Given the `root` of a binary tree, mirror it: at every node, swap the left and right children. Return the root of the mirrored tree.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [1,2,3,4,5]` → `[1,3,2,null,null,5,4]`\n\n**Example 2:** `root = [7,null,9]` → `[7,9]`\n\n**Constraints:**\n- The tree has `0` to `10^4` nodes\n- `-100 <= Node.val <= 100`",
    "description_zh": "给定二叉树的根节点 `root`，将整棵树镜像翻转：在每个节点上交换它的左子节点与右子节点。返回翻转后的根节点。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [1,2,3,4,5]` → `[1,3,2,null,null,5,4]`\n\n**示例 2：** `root = [7,null,9]` → `[7,9]`\n\n**约束：**\n- 节点数在 `0` 到 `10^4` 之间\n- `-100 <= Node.val <= 100`",
    "hint": "Inverting a tree means swapping the two children of every node. Do the swap at the current node, then invert both children.",
    "hint_zh": "翻转树就是交换每个节点的两个子节点。先在当前节点交换，再分别翻转两个子节点。",
    "cases": [
        {"args": [[1, 2, 3, 4, 5]], "expected": [1, 3, 2, None, None, 5, 4]},
        {"args": [[7, None, 9]], "expected": [7, 9]},
        {"args": [[]], "expected": []},
        {"args": [[1]], "expected": [1]},
        {"args": [[1, 2]], "expected": [1, None, 2]},
        # Swapping only at the root leaves the lower levels unswapped; here 5 must end up on the left
        {"args": [[1, 2, 3, 4, None, None, 5]], "expected": [1, 3, 2, 5, None, None, 4]},
        {"args": [[5, None, 8, 6, 9]], "expected": [5, 8, None, 9, 6]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is not None:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root''',
    "explanation": "Recursion. Mirroring a tree means swapping the two children of each node and mirroring each child subtree. Done top-down, every node is visited once and its children are swapped and then inverted.\n\nTime O(n), space O(h) for the recursion, where h is the height of the tree.",
    "explanation_zh": "递归。翻转树就是交换每个节点的两个子节点，并翻转每个子树。自顶向下做，每个节点恰好访问一次：交换它的子节点，再分别翻转。\n\n时间 O(n)，递归空间 O(h)，其中 h 为树的高度。",
}
