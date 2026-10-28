"""Kth Smallest Element in a BST (LeetCode 230)."""

import random

from torch_judge.algo.runner import TreeNode, tree_to_array


def _bst(values):
    # Insert values in the given order. The shape follows the order, so the cases vary
    root = None
    for v in values:
        node = TreeNode(v)
        if root is None:
            root = node
            continue
        cur = root
        while True:
            side = "left" if v < cur.val else "right"
            if getattr(cur, side) is None:
                setattr(cur, side, node)
                break
            cur = getattr(cur, side)
    return root


def _large_case():
    # 10^4 distinct values inserted in random order. An in-order walk is O(n), so no naive
    # approach times out at this size; the case checks scale and correctness
    rng = random.Random(230)
    values = rng.sample(range(-10**6, 10**6), 10000)
    k = rng.randint(1, len(values))
    return {"args": [tree_to_array(_bst(values)), k], "expected": sorted(values)[k - 1]}


def _random_case(rng):
    values = rng.sample(range(-30, 31), rng.randint(1, 10))
    return {"args": [tree_to_array(_bst(values)), rng.randint(1, len(values))]}


TASK = {
    "title": "Kth Smallest Element in a BST",
    "title_zh": "二叉搜索树中第 K 小的元素",
    "difficulty": "Medium",
    "leetcode_id": 230,
    "leetcode_slug": "kth-smallest-element-in-a-bst",
    "category": "Binary Tree",
    "function_name": "kthSmallest",
    "params": [("root", "TreeNode"), ("k", "int")],
    "returns": "int",
    "description_en": "Given the `root` of a binary search tree and an integer `k`, return the value that would sit at position `k` if every node's value were listed in ascending order. Positions start at 1, so `k = 1` gives the smallest value.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [6,3,8,1,4]`, `k = 2` → `3`\n\n**Example 2:** `root = [5,2,7,1,null,null,9]`, `k = 4` → `7`\n\n**Constraints:**\n- The tree has `1` to $10^{4}$ nodes, and $1 \\le \\texttt{k} \\le \\texttt{n}$, where `n` is the number of nodes\n- $-10^{6} \\le \\texttt{Node.val} \\le 10^{6}$\n- All node values are distinct\n\n**Follow-up:** if the tree changes often and you must answer many such queries, what extra information could each node store?",
    "description_zh": "给定一棵二叉搜索树的根节点 `root` 和整数 `k`，返回如果把所有节点的值按从小到大排列，位于第 `k` 个位置的值。位置从 1 开始计数，所以 `k = 1` 对应最小值。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [6,3,8,1,4]`，`k = 2` → `3`\n\n**示例 2：** `root = [5,2,7,1,null,null,9]`，`k = 4` → `7`\n\n**约束：**\n- 节点数在 `1` 到 $10^{4}$ 之间，且 $1 \\le \\texttt{k} \\le \\texttt{n}$，其中 `n` 为节点数\n- $-10^{6} \\le \\texttt{Node.val} \\le 10^{6}$\n- 所有节点的值互不相同\n\n**进阶：** 如果树经常被修改，而且需要反复回答这类查询，你会在每个节点上额外保存什么信息？",
    "hint": "An in-order walk visits the values in ascending order. Count the visits and stop at the k-th one.",
    "hint_zh": "中序遍历按从小到大的顺序访问节点的值。数一数访问的次数，到第 k 次时停下即可。",
    "cases": [
        {"args": [[6, 3, 8, 1, 4], 2], "expected": 3},
        {"args": [[5, 2, 7, 1, None, None, 9], 4], "expected": 7},
        {"args": [[1], 1], "expected": 1},
        {"args": [[2, 1], 1], "expected": 1},
        {"args": [[2, 1], 2], "expected": 2},
        # The answer is the root's left child's right child; a pre-order walk returns 3 here instead
        {"args": [[4, 2, 6, None, 3, 5, 7], 3], "expected": 4},
        {"args": [[10, 5, 15, 2, 8, 12, 20], 7], "expected": 20},
        # A left-leaning chain: pre-order gives 5 4 3 2 1, so the 5th visit would wrongly be 1
        {"args": [[5, 4, None, 3, None, 2, None, 1], 5], "expected": 5},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack, node = [], root
        while True:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            k -= 1
            if k == 0:
                return node.val
            node = node.right''',
    "explanation": "An in-order walk of a BST visits the values in ascending order. Walk with an explicit stack and count the visits; the k-th node visited is the answer, so the walk can stop there without touching the rest of the tree.\n\nTime O(h + k) in the worst case O(n), space O(h), where h is the height of the tree.",
    "explanation_zh": "二叉搜索树的中序遍历按从小到大的顺序访问节点的值。用显式栈做中序遍历并计数，第 k 个被访问的节点就是答案，遍历可以在那里停止，不必访问其余的节点。\n\n时间 O(h + k)，最坏为 O(n)，空间 O(h)，其中 h 为树的高度。",
}
