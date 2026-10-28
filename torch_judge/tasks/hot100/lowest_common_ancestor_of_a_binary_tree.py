"""Lowest Common Ancestor of a Binary Tree (LeetCode 236)."""

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
    # 10^5 nodes in a deep tree (about 6700 levels): a recursive search needs thousands of
    # frames, within the judge's recursion limit
    rng = random.Random(236)
    n = 100000
    vals = rng.sample(range(-10**9, 10**9), n)
    left, right = _random_shape(rng, n, window=40)
    parent = [-1] * n
    for i in range(n):
        for c in (left[i], right[i]):
            if c >= 0:
                parent[c] = i
    a, b = rng.sample(range(n), 2)
    # Ancestors of a (a included), then the first ancestor of b that is among them
    ancestors = set()
    x = a
    while x >= 0:
        ancestors.add(x)
        x = parent[x]
    y = b
    while y not in ancestors:
        y = parent[y]
    return {"args": [tree_to_array(_build(vals, left, right)), vals[a], vals[b]], "expected": vals[y]}


def _random_case(rng):
    n = rng.randint(2, 8)
    left, right = _random_shape(rng, n, window=rng.randint(1, 3))
    vals = rng.sample(range(-20, 21), n)
    p, q = rng.sample(vals, 2)
    return {"args": [tree_to_array(_build(vals, left, right)), p, q]}


TASK = {
    "title": "Lowest Common Ancestor of a Binary Tree",
    "title_zh": "二叉树的最近公共祖先",
    "difficulty": "Medium",
    "leetcode_id": 236,
    "leetcode_slug": "lowest-common-ancestor-of-a-binary-tree",
    "category": "Binary Tree",
    "function_name": "lowestCommonAncestor",
    "params": [("root", "TreeNode"), ("p", "TreeNode"), ("q", "TreeNode")],
    "returns": "TreeNode",
    # Cases give the tree and the two values; `prepare` finds the nodes, `finish` returns the value
    "inputs": [("root", "TreeNode"), ("p", "int"), ("q", "int")],
    "output": "int",
    "prepare": '''def prepare(args):
    root, p, q = args
    nodes = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node is not None:
            nodes[node.val] = node
            stack.append(node.left)
            stack.append(node.right)
    return [root, nodes[p], nodes[q]]''',
    "finish": '''def finish(result, args):
    # The answer must be a node of this very tree (checked by identity), not a look-alike
    stack = [args[0]]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if node is result:
            return result.val
        stack.append(node.left)
        stack.append(node.right)
    return "a node that is not in the tree"''',
    "description_en": "Given a binary tree and two of its nodes `p` and `q`, return their **lowest common ancestor** (LCA): the deepest node that has both `p` and `q` as descendants. A node counts as a descendant of itself, so when `p` is an ancestor of `q`, the answer is `p`.\n\nAll node values are distinct, and `p` and `q` are two different values that occur in the tree. Tests describe a case by the tree and the two values, and your method receives the matching node objects. Trees are written in level order, with `null` for a missing child, and the answer is the value of the LCA node.\n\n**Example 1:** `root = [10,4,15,2,7,null,20]`, `p = 2`, `q = 7` → `4`\n\n**Example 2:** `root = [10,4,15,2,7,null,20]`, `p = 4`, `q = 2` → `4` (4 is an ancestor of 2, so it is the answer)\n\n**Constraints:**\n- The tree has `2` to $10^{5}$ nodes\n- $-10^{9} \\le \\texttt{Node.val} \\le 10^{9}$, all values distinct\n- $\\texttt{p} \\ne \\texttt{q}$, and both occur in the tree",
    "description_zh": "给定一棵二叉树和其中的两个节点 `p`、`q`，返回它们的**最近公共祖先**（LCA）：即同时以 `p` 和 `q` 为后代的最深节点。节点可以是自身的后代，因此当 `p` 是 `q` 的祖先时，答案就是 `p`。\n\n所有节点值互不相同，`p` 和 `q` 是树中两个不同的值。测试用例以树和两个值来描述，你的方法接收到的是对应的节点对象。二叉树按层序写成数组，缺失的子节点写作 `null`；答案是 LCA 节点的值。\n\n**示例 1：** `root = [10,4,15,2,7,null,20]`，`p = 2`，`q = 7` → `4`\n\n**示例 2：** `root = [10,4,15,2,7,null,20]`，`p = 4`，`q = 2` → `4`（4 是 2 的祖先，所以答案是 4）\n\n**约束：**\n- 节点数在 `2` 到 $10^{5}$ 之间\n- $-10^{9} \\le \\texttt{Node.val} \\le 10^{9}$，所有值互不相同\n- $\\texttt{p} \\ne \\texttt{q}$，且两者都在树中",
    "hint": "One traversal records each node's parent. Walk up from p and mark every ancestor (p included). Then walk up from q: the first node you reach that is marked is the LCA.",
    "hint_zh": "先用一次遍历记录每个节点的父节点。从 p 向上走，标记沿途所有祖先（包括 p 自己）。再从 q 向上走，第一个被标记的节点就是 LCA。",
    "cases": [
        {"args": [[10, 4, 15, 2, 7, None, 20], 2, 7], "expected": 4},
        {"args": [[10, 4, 15, 2, 7, None, 20], 4, 2], "expected": 4},
        # The two nodes are in different subtrees of the root
        {"args": [[8, 3, 9, 1, 6, None, None, None, None, 4, 7], 4, 7], "expected": 6},
        {"args": [[8, 3, 9, 1, 6, None, None, None, None, 4, 7], 1, 7], "expected": 3},
        {"args": [[8, 3, 9, 1, 6, None, None, None, None, 4, 7], 9, 4], "expected": 8},
        # The root itself is one of the two nodes
        {"args": [[1, 2], 1, 2], "expected": 1},
        # A left-leaning chain: the deeper node is the answer's descendant, so the ancestor wins
        {"args": [[1, 2, None, 3, None, 4], 4, 2], "expected": 2},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def lowestCommonAncestor(self, root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
        # Parent of every node, found by one traversal. Iterative, since the tree can be very deep
        parent = {root: None}
        stack = [root]
        while stack:
            node = stack.pop()
            for child in (node.left, node.right):
                if child:
                    parent[child] = node
                    stack.append(child)
        # Every ancestor of p (p included); then climb from q to the first one of them
        ancestors = set()
        node = p
        while node:
            ancestors.add(node)
            node = parent[node]
        while q not in ancestors:
            q = parent[q]
        return q''',
    "explanation": "Record each node's parent with one traversal. Then mark the path from `p` up to the root, and climb from `q` until you reach a marked node. That node is the first common ancestor on the way up from `q`, which makes it the deepest one shared by `p` and `q`. A node counts as its own ancestor, so if `p` lies above `q`, the climb from `q` reaches `p` itself.\n\nThe usual recursive search is shorter, but this tree is tens of thousands of levels deep, far beyond Python's recursion limit, so this version walks the tree iteratively.\n\nTime O(n), space O(n).",
    "explanation_zh": "先用一次遍历记录每个节点的父节点。然后把从 `p` 到根的路径全部标记出来，再从 `q` 向上爬，直到遇到被标记的节点。这个节点就是从 `q` 往上第一个与 `p` 共享的祖先，也就是 `p` 和 `q` 共同的最深祖先。节点可以是自身的祖先，所以如果 `p` 在 `q` 的上方，从 `q` 向上爬时会直接遇到 `p`。\n\n常见的递归写法更简短，但这棵树的深度达到数万层，远超 Python 的递归上限，所以这里用迭代方式遍历。\n\n时间 O(n)，空间 O(n)。",
}
