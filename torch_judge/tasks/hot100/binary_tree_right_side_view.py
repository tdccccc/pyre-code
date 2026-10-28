"""Binary Tree Right Side View (LeetCode 199)."""

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
    # The constraint limit is 100 nodes, so every correct method is fast here; this case checks
    # the answer on a bushy random tree
    rng = random.Random(199)
    n = 100
    vals = [rng.randint(-100, 100) for _ in range(n)]
    left, right = _random_shape(rng, n, window=n)
    # Level by level, read left to right: the last node of each level is the visible one
    view, level = [], [0]
    while level:
        view.append(vals[level[-1]])
        level = [c for i in level for c in (left[i], right[i]) if c >= 0]
    return {"args": [tree_to_array(_build(vals, left, right))], "expected": view}


def _random_case(rng):
    n = rng.randint(0, 8)
    if n == 0:
        return {"args": [[]]}
    left, right = _random_shape(rng, n, window=rng.randint(1, 3))
    vals = [rng.randint(-9, 9) for _ in range(n)]
    return {"args": [tree_to_array(_build(vals, left, right))]}


TASK = {
    "title": "Binary Tree Right Side View",
    "title_zh": "二叉树的右视图",
    "difficulty": "Medium",
    "leetcode_id": 199,
    "leetcode_slug": "binary-tree-right-side-view",
    "category": "Binary Tree",
    "function_name": "rightSideView",
    "params": [("root", "TreeNode")],
    "returns": "int[]",
    "description_en": "Picture yourself standing on the right side of a binary tree and looking at it. Return the values you can see, from the top level down: for each level, the node furthest to the right on that level.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [4,2,9,null,1,7]` → `[4,9,7]` (the third level reads `1, 7` from left to right, so `7` is the one visible)\n\n**Example 2:** `root = [1,2]` → `[1,2]` (the root has only a left child, but that child is still the rightmost node of its level)\n\n**Constraints:**\n- The tree has `0` to `100` nodes\n- `-100 <= Node.val <= 100`",
    "description_zh": "想象你站在二叉树的右侧朝它看。返回从顶层到底层你能看到的节点值：每一层中最靠右的那个节点的值。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [4,2,9,null,1,7]` → `[4,9,7]`（第三层从左到右是 `1, 7`，能看到的是 7）\n\n**示例 2：** `root = [1,2]` → `[1,2]`（根节点只有左子节点，但它仍是所在层最靠右的节点）\n\n**约束：**\n- 节点数在 `0` 到 `100` 之间\n- `-100 <= Node.val <= 100`",
    "hint": "Breadth-first search, one level per round. The last node taken from the queue in each round is the rightmost one on that level, so its value is the one to keep.",
    "hint_zh": "用广度优先搜索逐层处理。每一轮从队列中最后取出的节点就是该层最右边的节点，保留它的值。",
    "cases": [
        {"args": [[4, 2, 9, None, 1, 7]], "expected": [4, 9, 7]},
        {"args": [[1, 2]], "expected": [1, 2]},
        {"args": [[]], "expected": []},
        {"args": [[1]], "expected": [1]},
        # A left-leaning chain: each level has one node, and it is the one visible
        {"args": [[1, 2, None, 3, None, 4]], "expected": [1, 2, 3, 4]},
        # The only node on level 2 sits under the left side. A left-first walk that keeps the
        # first node it reaches per depth would report 2 for level 1 and 4 for level 2
        {"args": [[1, 2, 3, 4]], "expected": [1, 3, 4]},
        {"args": [[0, -5, None, None, -7]], "expected": [0, -5, -7]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import deque


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        view, queue = [], deque([root])
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            # Queue order is left to right, so the last node of the round is the rightmost one
            view.append(node.val)
        return view''',
    "explanation": "Level-order BFS. At the start of each round the queue holds exactly one level, in left-to-right order, so the last node taken in that round is the rightmost node of the level, and its value goes into the answer. A depth-first walk that visits the right subtree first and records the first node seen at each depth works too.\n\nTime O(n), space O(w), where w is the width of the widest level.",
    "explanation_zh": "层序 BFS。每一轮开始时队列恰好是一整层，且按从左到右排列，所以本轮最后取出的节点就是该层最右边的节点，把它的值放进答案。先访问右子树的深度优先遍历，并记录每个深度第一次遇到的节点，也同样正确。\n\n时间 O(n)，空间 O(w)，w 为最宽一层的节点数。",
}
