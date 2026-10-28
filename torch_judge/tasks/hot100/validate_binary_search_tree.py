"""Validate Binary Search Tree (LeetCode 98)."""

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


def _balanced(values):
    # Balanced tree from a sorted list, used to build the deep test tree
    if not values:
        return None
    mid = len(values) // 2
    return TreeNode(values[mid], _balanced(values[:mid]), _balanced(values[mid + 1:]))


def _is_valid(root):
    # Each node must lie strictly inside the bounds set by its ancestors; explicit stack
    stack = [(root, -10**18, 10**18)]
    while stack:
        node, lo, hi = stack.pop()
        if node is None:
            continue
        if not lo < node.val < hi:
            return False
        stack.append((node.left, lo, node.val))
        stack.append((node.right, node.val, hi))
    return True


def _large_case():
    # A valid, deep BST: a 9000-node right-leaning spine whose last node's left child is a
    # balanced BST of 1000 values, all between the spine's last two values. A naive "check each
    # subtree" pass costs about 5 * 10^7 steps (several seconds). The answer is true, so that
    # pass cannot stop early. The reference is iterative; recursive ACM programs need a raised
    # recursion limit at this depth
    rng = random.Random(98)
    spine = [TreeNode(i * 10**5) for i in range(9000)]
    for a, b in zip(spine, spine[1:]):
        a.right = b
    lo, hi = spine[-2].val, spine[-1].val
    spine[-1].left = _balanced(sorted(rng.sample(range(lo + 1, hi), 1000)))
    root = spine[0]
    return {"args": [tree_to_array(root)], "expected": _is_valid(root)}


def _random_case(rng):
    # Half the cases are valid BSTs; the other half get one node moved to a random value
    root = _bst(rng.sample(range(-9, 10), rng.randint(1, 10)))
    if rng.random() < 0.5:
        nodes = [root]
        for node in nodes:
            nodes += [c for c in (node.left, node.right) if c is not None]
        rng.choice(nodes).val = rng.randint(-9, 9)
    return {"args": [tree_to_array(root)]}


TASK = {
    "title": "Validate Binary Search Tree",
    "title_zh": "验证二叉搜索树",
    "difficulty": "Medium",
    "leetcode_id": 98,
    "leetcode_slug": "validate-binary-search-tree",
    "category": "Binary Tree",
    "function_name": "isValidBST",
    "params": [("root", "TreeNode")],
    "returns": "bool",
    "description_en": "Given the `root` of a binary tree, decide whether it is a valid binary search tree. In a valid one, for every node, every value in its left subtree is **strictly smaller** than the node's value, and every value in its right subtree is **strictly larger**. Equal values are not allowed.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [10,5,15,null,null,6,20]` → `false` (6 sits in the right subtree of 10, yet it is smaller than 10)\n\n**Example 2:** `root = [8,3,9,1,5]` → `true`\n\n**Constraints:**\n- The tree has `1` to `10^4` nodes\n- `-10^9 <= Node.val <= 10^9`",
    "description_zh": "给定二叉树的根节点 `root`，判断它是不是一棵合法的二叉搜索树。合法的条件是：对于每个节点，左子树中的所有值都**严格小于**该节点的值，右子树中的所有值都**严格大于**该节点的值。不允许出现相等的值。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [10,5,15,null,null,6,20]` → `false`（6 位于 10 的右子树中，但它比 10 小）\n\n**示例 2：** `root = [8,3,9,1,5]` → `true`\n\n**约束：**\n- 节点数在 `1` 到 `10^4` 之间\n- `-10^9 <= Node.val <= 10^9`",
    "hint": "Give every node an open interval of allowed values, inherited from its ancestors. Going left tightens the upper bound to the parent's value, going right tightens the lower bound.",
    "hint_zh": "为每个节点维护一个允许取值的开区间，它从祖先继承而来。向左走时把上界收紧为父节点的值，向右走时把下界收紧为父节点的值。",
    "cases": [
        {"args": [[10, 5, 15, None, None, 6, 20]], "expected": False},
        {"args": [[8, 3, 9, 1, 5]], "expected": True},
        {"args": [[1]], "expected": True},
        # Equal values are not allowed; a check written with <= would accept this
        {"args": [[2, 2]], "expected": False},
        {"args": [[4, 2, 6, 1, 3]], "expected": True},
        # 9 is in the left subtree of 7 (under 4). Every parent and child pair still looks fine
        {"args": [[7, 4, 12, 2, 9]], "expected": False},
        # The 5 under 3 equals the root 5, an ancestor, not its parent
        {"args": [[5, 3, 7, 1, 5]], "expected": False},
        {"args": [[20, 10, 30, 5, 15, 25, 35]], "expected": True},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = [(root, float("-inf"), float("inf"))]
        while stack:
            node, lo, hi = stack.pop()
            if node is None:
                continue
            if not lo < node.val < hi:
                return False
            stack.append((node.left, lo, node.val))
            stack.append((node.right, node.val, hi))
        return True''',
    "explanation": "Carry the open interval of allowed values down the tree. The root may hold anything. A left child must be below its parent and still above the lower bound inherited from above, and a right child is symmetric. Checking every node against its inherited interval catches violations from any ancestor, not only the parent. Checking only parent and child is not enough.\n\nTime O(n), space O(h) for the stack in the worst case.",
    "explanation_zh": "沿树向下传递允许取值的开区间。根节点可以是任意值。左子节点必须小于父节点，并且大于从祖先继承来的下界；右子节点同理。把每个节点都和它继承来的区间比较，就能发现任意祖先造成的违规，而不只是父子之间的违规。只比较父子关系是不够的。\n\n时间 O(n)，最坏情况下栈空间 O(h)。",
}
