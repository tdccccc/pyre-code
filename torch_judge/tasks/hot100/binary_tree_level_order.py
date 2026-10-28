"""Binary Tree Level Order Traversal (LeetCode 102)."""

import random

from torch_judge.algo.runner import TreeNode, tree_to_array


def _large_case():
    # A random tree of 2000 nodes, encoded level by level
    rng = random.Random(102)
    values = [rng.randint(-1000, 1000)]
    levels = [[values[0]]]
    frontier = 1
    remaining = 1999
    while remaining > 0:
        level = []
        for _ in range(frontier):
            for _ in range(2):
                if remaining > 0 and rng.random() < 0.8:
                    v = rng.randint(-1000, 1000)
                    values.append(v)
                    level.append(v)
                    remaining -= 1
                else:
                    values.append(None)
        if not level:
            # Every child came up empty; force one so the tree keeps growing
            values[-1] = rng.randint(-1000, 1000)
            level.append(values[-1])
            remaining -= 1
        levels.append(level)
        frontier = len(level)
    while values[-1] is None:
        values.pop()
    return {"args": [values], "expected": levels}


def _random_case(rng):
    # Insert distinct values into a BST: random shapes, including lopsided ones
    values = rng.sample(range(-20, 21), rng.randint(0, 10))
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
    return {"args": [tree_to_array(root)]}


TASK = {
    "title": "Binary Tree Level Order Traversal",
    "title_zh": "二叉树的层序遍历",
    "difficulty": "Medium",
    "leetcode_id": 102,
    "leetcode_slug": "binary-tree-level-order-traversal",
    "category": "Binary Tree",
    "function_name": "levelOrder",
    "params": [("root", "TreeNode")],
    "returns": "int[][]",
    "description_en": "Given the root `root` of a binary tree, return the node values level by level: the first list holds the root, the next holds its children, and so on, each level read left to right.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [5,3,8,null,4,7,9]` → `[[5],[3,8],[4,7,9]]`\n\n**Example 2:** `root = [1]` → `[[1]]`\n\n**Constraints:**\n- The tree has `0` to `2000` nodes\n- $-1000 \\le \\texttt{Node.val} \\le 1000$",
    "description_zh": "给定二叉树的根节点 `root`，按层返回节点值：第一个列表是根节点，下一个是它的子节点，依此类推，每层从左到右。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [5,3,8,null,4,7,9]` → `[[5],[3,8],[4,7,9]]`\n\n**示例 2：** `root = [1]` → `[[1]]`\n\n**约束：**\n- 节点数在 `0` 到 `2000` 之间\n- $-1000 \\le \\texttt{Node.val} \\le 1000$",
    "hint": "Breadth-first search with a queue. At the start of each round, the queue holds exactly one level: pop `len(queue)` nodes, record their values, and push their children.",
    "hint_zh": "用队列做广度优先搜索。每一轮开始时，队列里恰好是一整层：弹出 `len(queue)` 个节点，记录它们的值，再把它们的子节点入队。",
    "cases": [
        {"args": [[5, 3, 8, None, 4, 7, 9]], "expected": [[5], [3, 8], [4, 7, 9]]},
        {"args": [[1]], "expected": [[1]]},
        {"args": [[]], "expected": []},
        {"args": [[1, 2, None, 3, None, 4]], "expected": [[1], [2], [3], [4]]},
        {"args": [[0, -1, 1, -2, None, None, 2]], "expected": [[0], [-1, 1], [-2, 2]]},
        {"args": [[1, None, 2, None, 3]], "expected": [[1], [2], [3]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import deque


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        result, queue = [], deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result''',
    "explanation": "BFS, one level per round. The queue's length at the start of a round is the size of the current level, so popping exactly that many nodes separates the levels cleanly while their children are queued for the next round.\n\nTime O(n), space O(width of the widest level).",
    "explanation_zh": "BFS，每轮处理一层。每轮开始时队列的长度就是当前层的节点数，恰好弹出这么多个节点，就能把各层干净地分开，同时把子节点排进下一轮。\n\n时间 O(n)，空间 O(最宽一层的宽度)。",
}
