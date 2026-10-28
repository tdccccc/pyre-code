"""Binary Tree Inorder Traversal (LeetCode 94)."""

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


def _inorder(root):
    # Explicit-stack reference for the expected answer, independent of the solution
    out, stack, node = [], [], root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def _large_case():
    # 10^4 nodes. The intended O(n) cannot time out at this size, so this checks scale only
    rng = random.Random(94)
    root = _random_tree(rng, 10000, -1000, 1000)
    return {"args": [tree_to_array(root)], "expected": _inorder(root)}


def _random_case(rng):
    return {"args": [tree_to_array(_random_tree(rng, rng.randint(0, 8), -9, 9))]}


TASK = {
    "title": "Binary Tree Inorder Traversal",
    "title_zh": "二叉树的中序遍历",
    "difficulty": "Easy",
    "leetcode_id": 94,
    "leetcode_slug": "binary-tree-inorder-traversal",
    "category": "Binary Tree",
    "function_name": "inorderTraversal",
    "params": [("root", "TreeNode")],
    "returns": "int[]",
    "description_en": "Given the `root` of a binary tree, return the values of its nodes in **in-order** order: first everything in the left subtree, then the node itself, then everything in the right subtree.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `root = [9,4,7,null,5,2]` → `[4,5,9,2,7]`\n\n**Example 2:** `root = [6]` → `[6]`\n\n**Constraints:**\n- The tree has `0` to `10^4` nodes\n- `-1000 <= Node.val <= 1000`\n\n**Follow-up:** can you write it with an explicit stack instead of recursion?",
    "description_zh": "给定二叉树的根节点 `root`，返回它所有节点值的**中序**遍历结果：先遍历左子树，再访问节点本身，最后遍历右子树。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `root = [9,4,7,null,5,2]` → `[4,5,9,2,7]`\n\n**示例 2：** `root = [6]` → `[6]`\n\n**约束：**\n- 节点数在 `0` 到 `10^4` 之间\n- `-1000 <= Node.val <= 1000`\n\n**进阶：** 你能不用递归，改用显式的栈来实现吗？",
    "hint": "Walk down the left spine, pushing each node on a stack. Pop a node, record it, then continue from its right child.",
    "hint_zh": "沿着左侧一路向下，把经过的节点依次压入栈中。弹出一个节点并记录它的值，然后从它的右子节点继续。栈里保存的是还没轮到访问的祖先节点。",
    "cases": [
        {"args": [[9, 4, 7, None, 5, 2]], "expected": [4, 5, 9, 2, 7]},
        {"args": [[6]], "expected": [6]},
        {"args": [[]], "expected": []},
        # The node must come after its left child; visiting it first (pre-order) gives [2, 1]
        {"args": [[1, 2]], "expected": [2, 1]},
        # Left-leaning chain: pre-order would give 5 4 3 2 1
        {"args": [[5, 4, None, 3, None, 2, None, 1]], "expected": [1, 2, 3, 4, 5]},
        {"args": [[3, None, 8, 2, None, 5]], "expected": [3, 5, 2, 8]},
        # Right-leaning chain: a stack walk that never steps to the right child stops early
        {"args": [[1, None, 2, None, 3]], "expected": [1, 2, 3]},
        {"args": [[-1, -2, 4, None, -3, 0, 5]], "expected": [-2, -3, -1, 0, 4, 5]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result, stack, node = [], [], root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            result.append(node.val)
            node = node.right
        return result''',
    "explanation": "Iterative inorder walk with an explicit stack. Every node first goes onto the stack while we descend its left spine, so its left subtree is finished before it is popped. Popping records the value, and then the walk continues in the right subtree. Recursion does the same thing with the call stack.\n\nTime O(n), space O(h) for the stack, where h is the height of the tree.",
    "explanation_zh": "用显式栈实现的迭代中序遍历。沿左侧一路向下时，每个节点先入栈，所以它的左子树会在它被弹出之前全部处理完。弹出时记录节点的值，然后转向右子树。递归做的是同样的事情，只是用的是调用栈。\n\n时间 O(n)，栈空间 O(h)，其中 h 为树的高度。",
}
