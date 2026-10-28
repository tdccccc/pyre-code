"""Symmetric Tree (LeetCode 101)."""

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


def _mirror(node):
    # A mirrored copy of a subtree
    if node is None:
        return None
    return TreeNode(node.val, _mirror(node.right), _mirror(node.left))


def _is_symmetric(root):
    # Compare pairs of positions across the root with an explicit stack
    pairs = [(root.left, root.right)]
    while pairs:
        a, b = pairs.pop()
        if a is None or b is None:
            if a is not b:
                return False
            continue
        if a.val != b.val:
            return False
        pairs.append((a.left, b.right))
        pairs.append((a.right, b.left))
    return True


def _large_case():
    # 999 nodes: a random half, its mirror image, then one value changed deep in the right
    # half so the answer is false. A value-only comparison can't tell the halves apart here
    rng = random.Random(101)
    left = _random_tree(rng, 499, 1, 3)
    root = TreeNode(2, left, _mirror(left))
    right_nodes = [root.right]
    for node in right_nodes:
        right_nodes += [c for c in (node.left, node.right) if c is not None]
    rng.choice(right_nodes).val = 99
    return {"args": [tree_to_array(root)], "expected": _is_symmetric(root)}


def _random_case(rng):
    # Half the time the right half is the mirror of the left, so both answers are common
    left = _random_tree(rng, rng.randint(0, 4), 1, 3)
    right = _mirror(left) if rng.random() < 0.5 else _random_tree(rng, rng.randint(0, 4), 1, 3)
    return {"args": [tree_to_array(TreeNode(2, left, right))]}


TASK = {
    "title": "Symmetric Tree",
    "title_zh": "对称二叉树",
    "difficulty": "Easy",
    "leetcode_id": 101,
    "leetcode_slug": "symmetric-tree",
    "category": "Binary Tree",
    "function_name": "isSymmetric",
    "params": [("root", "TreeNode")],
    "returns": "bool",
    "description_en": "Given the `root` of a binary tree, decide whether the tree is symmetric: the left subtree must be a mirror image of the right subtree, so the two halves reflect each other across the root.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [6,8,8,null,5,5,null]` → `true`\n\n**Example 2:** `root = [6,8,8,null,5,null,5]` → `false`\n\n**Constraints:**\n- The tree has `1` to `1000` nodes\n- `-100 <= Node.val <= 100`\n\n**Follow-up:** can you solve it by comparing pairs of nodes level by level with a queue?",
    "description_zh": "给定二叉树的根节点 `root`，判断它是否轴对称：左子树必须是右子树的镜像，两半以根节点为轴互相映照。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [6,8,8,null,5,5,null]` → `true`\n\n**示例 2：** `root = [6,8,8,null,5,null,5]` → `false`\n\n**约束：**\n- 节点数在 `1` 到 `1000` 之间\n- `-100 <= Node.val <= 100`\n\n**进阶：** 你能用队列逐层比较成对的节点来解决吗？",
    "hint": "Two subtrees mirror each other when their roots hold the same value, the left one's left child mirrors the right one's right child, and the left one's right child mirrors the right one's left child. Recurse on those pairs.",
    "hint_zh": "两棵子树互为镜像，当且仅当两个根的值相同，左树的左子节点与右树的右子节点互为镜像，且左树的右子节点与右树的左子节点互为镜像。对这些配对递归即可。",
    "cases": [
        {"args": [[6, 8, 8, None, 5, 5, None]], "expected": True},
        {"args": [[6, 8, 8, None, 5, None, 5]], "expected": False},
        {"args": [[1]], "expected": True},
        {"args": [[1, 2, 2]], "expected": True},
        # Same in-order values read as a palindrome (2 3 1 3 2), but the two halves have different shapes
        {"args": [[1, 2, 3, None, 3, None, 2]], "expected": False},
        # Equal values in mirrored inner positions
        {"args": [[1, 2, 2, None, 3, 3, None]], "expected": True},
        # Outer pairs match, nothing else needs to
        {"args": [[3, 1, 1, 2, None, None, 2]], "expected": True},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def mirror(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
            if a is None or b is None:
                return a is b
            return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)

        return mirror(root.left, root.right)''',
    "explanation": "Compare the two halves as mirror images. Two trees mirror each other when both are empty, or when both are non-empty with equal values, the left child of one mirroring the right child of the other, and the right child of one mirroring the left child of the other. Start the comparison with the root's two children.\n\nTime O(n), space O(h) for the recursion, where h is the height of the tree.",
    "explanation_zh": "把左右两半当作镜像来比较。两棵树互为镜像，当且仅当两者都为空，或者两者都非空、值相等，并且一棵树的左子节点与另一棵树的右子节点互为镜像，右子节点与左子节点也互为镜像。比较从根节点的两个子节点开始。\n\n时间 O(n)，递归空间 O(h)，其中 h 为树的高度。",
}
