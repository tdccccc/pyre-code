"""Flatten Binary Tree to Linked List (LeetCode 114)."""

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
    # 2000 nodes forming a deep tree (over 1000 levels): a recursive flatten needs that many
    # frames, beyond Python's default of 1000 but within the judge's limit of 20000
    rng = random.Random(114)
    n = 2000
    vals = [rng.randint(-100, 100) for _ in range(n)]
    left, right = _random_shape(rng, n, window=3)
    order, stack = [], [0]  # preorder of the original tree
    while stack:
        i = stack.pop()
        order.append(vals[i])
        if right[i] >= 0:
            stack.append(right[i])
        if left[i] >= 0:
            stack.append(left[i])
    # Flattened: each value, then the empty left slot of its node (no trailing null)
    expected = [x for v in order for x in (v, None)][:-1]
    return {"args": [tree_to_array(_build(vals, left, right))], "expected": expected}


def _random_case(rng):
    n = rng.randint(0, 8)
    if n == 0:
        return {"args": [[]]}
    left, right = _random_shape(rng, n, window=rng.randint(1, 3))
    vals = [rng.randint(-9, 9) for _ in range(n)]
    return {"args": [tree_to_array(_build(vals, left, right))]}


TASK = {
    "title": "Flatten Binary Tree to Linked List",
    "title_zh": "二叉树展开为链表",
    "difficulty": "Medium",
    "leetcode_id": 114,
    "leetcode_slug": "flatten-binary-tree-to-linked-list",
    "category": "Binary Tree",
    "function_name": "flatten",
    "params": [("root", "TreeNode")],
    "returns": "void",
    "inplace": "root",
    "description_en": "Rearrange a binary tree **in place** into a chain. Every node's `right` pointer leads to the next node of the chain, every `left` pointer is `null`, and the chain follows the preorder of the original tree: a node, then its whole left subtree, then its whole right subtree.\n\nDo not return anything; change the tree that `root` points to.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [3,1,4,null,2]` → `[3,null,1,null,2,null,4]` (preorder `3, 1, 2, 4`)\n\n**Example 2:** `root = [0]` → `[0]`\n\n**Constraints:**\n- The tree has `0` to `2000` nodes\n- `-100 <= Node.val <= 100`\n\n**Follow-up:** can you flatten it without recursion and using O(1) extra space?",
    "description_zh": "原地把二叉树改造成一条链表：每个节点的 `right` 指针指向链表中的下一个节点，所有 `left` 指针都为 `null`，链表的顺序与原树的前序遍历一致：先是一个节点，接着是它的整个左子树，然后是它的整个右子树。\n\n不要返回任何值，直接修改 `root` 指向的树。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [3,1,4,null,2]` → `[3,null,1,null,2,null,4]`（前序为 `3, 1, 2, 4`）\n\n**示例 2：** `root = [0]` → `[0]`\n\n**约束：**\n- 节点数在 `0` 到 `2000` 之间\n- `-100 <= Node.val <= 100`\n\n**进阶：** 你能不用递归、只用 O(1) 的额外空间完成吗？",
    "hint": "Walk down the right chain. When a node has a left subtree, find that subtree's rightmost node, hang the old right subtree from it, and move the left subtree into the right slot.",
    "hint_zh": "沿着右链向下走。遇到有左子树的节点时，找到左子树最右边的节点，把原来的右子树挂在它下面，再把左子树整体移到右指针的位置。",
    "cases": [
        {"args": [[3, 1, 4, None, 2]], "expected": [3, None, 1, None, 2, None, 4]},
        {"args": [[0]], "expected": [0]},
        {"args": [[]], "expected": []},
        {"args": [[1, 2]], "expected": [1, None, 2]},
        {"args": [[1, None, 2, 3]], "expected": [1, None, 2, None, 3]},
        # Overwriting `right` before saving the old right subtree loses 8 and 9
        {"args": [[5, 2, 8, 1, None, 7, 9]], "expected": [5, None, 2, None, 1, None, 8, None, 7, None, 9]},
        {"args": [[4, 3, None, 2, None, 1]], "expected": [4, None, 3, None, 2, None, 1]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        cur = root
        while cur:
            if cur.left:
                # In preorder the left subtree comes right after cur, and the right subtree
                # after the left one. So hang cur.right from the left subtree's last node.
                pre = cur.left
                while pre.right:
                    pre = pre.right
                pre.right = cur.right
                cur.right = cur.left
                cur.left = None
            cur = cur.right''',
    "explanation": "Walk down the chain one node at a time. If the current node has a left subtree, that subtree has to sit between the node and its right subtree in preorder. The last node of the left subtree's preorder is its rightmost node, so the right subtree is attached there, the left subtree is moved into the right slot, and the left pointer is cleared. Nothing is recursive, and the work per node is constant apart from the walk down the left subtree's right spine, which each node is part of only a constant number of times.\n\nTime O(n), space O(1).",
    "explanation_zh": "沿着链一个节点一个节点地处理。如果当前节点有左子树，那么在前序中左子树应当位于该节点和右子树之间。左子树前序遍历的最后一个节点就是它最右边的节点，于是把原来的右子树挂在那里，把左子树整体移到右指针的位置，并把左指针清空。整个过程没有递归；除了沿左子树的右链向下走之外，每个节点只做常数次操作，而每个节点被走过的次数也是常数。\n\n时间 O(n)，空间 O(1)。",
}
