"""Construct Binary Tree from Preorder and Inorder Traversal (LeetCode 105)."""

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


def _orders(vals, left, right):
    """Preorder and inorder value lists of the tree. Iterative, so deep trees are fine."""
    pre, stack = [], [0]
    while stack:
        i = stack.pop()
        pre.append(vals[i])
        if right[i] >= 0:
            stack.append(right[i])
        if left[i] >= 0:
            stack.append(left[i])
    ino, stack, i = [], [], 0
    while stack or i >= 0:
        while i >= 0:
            stack.append(i)
            i = left[i]
        i = stack.pop()
        ino.append(vals[i])
        i = right[i]
    return pre, ino


def _large_case():
    # A zigzag path of 3000 nodes, 3000 levels deep: a recursive build needs that many frames
    # (the judge allows 20000). Searching inorder for each root with list.index is O(n^2) but
    # still takes well under a second at n <= 3000, so this checks correctness at scale
    rng = random.Random(105)
    n = 3000
    vals = rng.sample(range(-3000, 3001), n)
    left, right = _random_shape(rng, n, window=2)
    preorder, inorder = _orders(vals, left, right)
    return {"args": [preorder, inorder], "expected": tree_to_array(_build(vals, left, right))}


def _random_case(rng):
    n = rng.randint(1, 8)
    left, right = _random_shape(rng, n, window=rng.randint(1, 3))
    vals = rng.sample(range(-20, 21), n)
    preorder, inorder = _orders(vals, left, right)
    return {"args": [preorder, inorder]}


TASK = {
    "title": "Construct Binary Tree from Preorder and Inorder Traversal",
    "title_zh": "从前序与中序遍历序列构造二叉树",
    "difficulty": "Medium",
    "leetcode_id": 105,
    "leetcode_slug": "construct-binary-tree-from-preorder-and-inorder-traversal",
    "category": "Binary Tree",
    "function_name": "buildTree",
    "params": [("preorder", "int[]"), ("inorder", "int[]")],
    "returns": "TreeNode",
    "description_en": "Given two integer arrays `preorder` and `inorder`, the preorder and inorder traversals of the same binary tree, rebuild the tree and return its root. All values are distinct.\n\nPreorder visits a node, then its left subtree, then its right subtree. Inorder visits the left subtree, then the node, then the right subtree.\n\nThe judge reads the tree you return in level order, with `null` for a missing child.\n\n**Example 1:** `preorder = [5,1,4,8,6]`, `inorder = [1,5,8,4,6]` → `[5,1,4,null,null,8,6]`\n\n**Example 2:** `preorder = [9,4,2]`, `inorder = [2,4,9]` → `[9,4,null,2]`\n\n**Constraints:**\n- $1 \\le \\texttt{preorder.length} \\le 3000$\n- `inorder.length == preorder.length`\n- $-3000 \\le \\texttt{preorder[i]},\\ \\texttt{inorder[i]} \\le 3000$\n- All values are distinct, and both arrays describe the same tree",
    "description_zh": "给定两个整数数组 `preorder` 和 `inorder`，它们分别是同一棵二叉树的前序遍历和中序遍历。请还原这棵树并返回它的根节点。所有值互不相同。\n\n前序遍历的顺序是：节点、左子树、右子树。中序遍历的顺序是：左子树、节点、右子树。\n\n判题程序会按层序读取你返回的树，缺失的子节点写作 `null`。\n\n**示例 1：** `preorder = [5,1,4,8,6]`，`inorder = [1,5,8,4,6]` → `[5,1,4,null,null,8,6]`\n\n**示例 2：** `preorder = [9,4,2]`，`inorder = [2,4,9]` → `[9,4,null,2]`\n\n**约束：**\n- $1 \\le \\texttt{preorder.length} \\le 3000$\n- `inorder.length == preorder.length`\n- $-3000 \\le \\texttt{preorder[i]},\\ \\texttt{inorder[i]} \\le 3000$\n- 所有值互不相同，两个数组描述的是同一棵树",
    "hint": "The first preorder value is the root. Its position in inorder splits the rest: the values before it form the left subtree, the values after it the right subtree, and preorder lists the left subtree's values before the right one's.",
    "hint_zh": "前序的第一个值是根节点。它在中序中的位置把其余部分分成两段：左边是左子树，右边是右子树；而前序中左子树的值排在右子树之前。",
    "cases": [
        {"args": [[5, 1, 4, 8, 6], [1, 5, 8, 4, 6]], "expected": [5, 1, 4, None, None, 8, 6]},
        {"args": [[9, 4, 2], [2, 4, 9]], "expected": [9, 4, None, 2]},
        {"args": [[7], [7]], "expected": [7]},
        # A right-leaning chain: the left subtree of every node is empty
        {"args": [[1, 2, 3], [1, 2, 3]], "expected": [1, None, 2, None, 3]},
        # Splitting preorder at the wrong place mixes up the two subtrees of 3
        {"args": [[3, 1, 2, 4], [1, 2, 3, 4]], "expected": [3, 1, 4, None, 2]},
        {"args": [[1, 2, 3, 4, 5], [2, 3, 1, 5, 4]], "expected": [1, 2, 4, None, 3, 5]},
        {"args": [[-1, -2, -3], [-3, -2, -1]], "expected": [-1, -2, None, -3]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        position = {v: i for i, v in enumerate(inorder)}  # value -> index in inorder
        next_value = iter(preorder)  # consumed root, then left subtree, then right subtree

        def build(lo: int, hi: int) -> Optional[TreeNode]:
            # The subtree whose inorder values are inorder[lo..hi]
            if lo > hi:
                return None
            root = TreeNode(next(next_value))
            mid = position[root.val]
            root.left = build(lo, mid - 1)
            root.right = build(mid + 1, hi)
            return root

        return build(0, len(inorder) - 1)''',
    "explanation": "The first value of `preorder` is always the root. Find it in `inorder`: the values before it form the left subtree and the values after it form the right subtree. Preorder lists the left subtree's values right after the root and the right subtree's values after those, so one iterator through `preorder` supplies the roots in the right order. A dict from value to inorder index makes each split O(1). The recursion is as deep as the tree, so the solution raises the recursion limit for skewed trees.\n\nTime O(n), space O(n).",
    "explanation_zh": "`preorder` 的第一个值总是根节点。在 `inorder` 中找到它：它左边的值组成左子树，右边的值组成右子树。前序中根节点之后紧接着是左子树的值，然后才是右子树的值，所以用一个沿 `preorder` 前进的迭代器，就能按正确的顺序依次取出每个子树的根。用字典记录 值 → 中序下标，每次划分都是 O(1)。递归深度与树的深度相同，所以遇到退化的树时要调大递归上限。\n\n时间 O(n)，空间 O(n)。",
}
