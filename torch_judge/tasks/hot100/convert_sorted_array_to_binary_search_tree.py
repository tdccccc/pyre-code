"""Convert Sorted Array to Binary Search Tree (LeetCode 108)."""

import random

from torch_judge.algo.runner import TreeNode, tree_to_array


def _balanced(values):
    # Midpoint recursion: the root is the middle value, so both halves differ in size by at most 1
    if not values:
        return None
    mid = len(values) // 2
    return TreeNode(values[mid], _balanced(values[:mid]), _balanced(values[mid + 1:]))


def _large_case():
    # 10^4 strictly increasing values. Any balanced answer is accepted, so the checker verifies
    # the result instead of comparing it with this expected tree
    rng = random.Random(108)
    nums = sorted(rng.sample(range(-10**5, 10**5), 10000))
    return {"args": [nums], "expected": tree_to_array(_balanced(nums))}


def _random_case(rng):
    return {"args": [sorted(rng.sample(range(-20, 21), rng.randint(1, 10)))]}


TASK = {
    "title": "Convert Sorted Array to Binary Search Tree",
    "title_zh": "将有序数组转换为二叉搜索树",
    "difficulty": "Easy",
    "leetcode_id": 108,
    "leetcode_slug": "convert-sorted-array-to-binary-search-tree",
    "category": "Binary Tree",
    "function_name": "sortedArrayToBST",
    "params": [("nums", "int[]")],
    "returns": "TreeNode",
    # Several trees are valid, so the checker accepts any balanced BST holding exactly nums
    "check": "custom",
    "checker": '''def check(args, actual, expected):
    values = []

    def height(node):
        if node is None:
            return 0
        left = height(node.left)
        values.append(node.val)
        right = height(node.right)
        if left < 0 or right < 0 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)

    return height(build_tree(actual)) >= 0 and values == args[0]''',
    "description_en": "You are given an integer array `nums` sorted in strictly increasing order. Build a **height-balanced** binary search tree that holds exactly these values, and return its root. In a height-balanced tree, the two subtrees of every node have heights that differ by at most 1. Several trees can qualify, and any one that meets both conditions is accepted.\n\nTrees are written in level order, with `null` for a missing child.\n\n**Example 1:** `nums = [-4,-1,2,6]` → `[-1,-4,2,null,null,null,6]` (one valid answer; `[2,-1,6,-4]` is another)\n\n**Example 2:** `nums = [7]` → `[7]`\n\n**Constraints:**\n- `1 <= nums.length <= 10^4`\n- `-10^5 <= nums[i] <= 10^5`\n- `nums` is sorted in strictly increasing order",
    "description_zh": "给定一个按严格递增顺序排列的整数数组 `nums`。请构造一棵**高度平衡**的二叉搜索树，恰好包含数组中的全部值，并返回它的根节点。高度平衡指：每个节点的左子树与右子树的高度差都不超过 1。满足条件的树可能不止一棵，任何一棵合格的树都会被接受。\n\n二叉树按层序写成数组，缺失的子节点写作 `null`。\n\n**示例 1：** `nums = [-4,-1,2,6]` → `[-1,-4,2,null,null,null,6]`（一个合格的答案；`[2,-1,6,-4]` 也是）\n\n**示例 2：** `nums = [7]` → `[7]`\n\n**约束：**\n- `1 <= nums.length <= 10^4`\n- `-10^5 <= nums[i] <= 10^5`\n- `nums` 按严格递增顺序排列",
    "hint": "Pick the middle element as the root. Everything to its left becomes the left subtree and everything to its right becomes the right subtree, and both halves are built the same way.",
    "hint_zh": "取中间的元素作为根节点。它左边的元素构成左子树，右边的元素构成右子树，两半用同样的方法递归构造。",
    "cases": [
        {"args": [[-4, -1, 2, 6]], "expected": [-1, -4, 2, None, None, None, 6]},
        {"args": [[7]], "expected": [7]},
        {"args": [[3, 8]], "expected": [3, None, 8]},
        # Taking the first element as the root gives a right-leaning chain; three values are enough to fail
        {"args": [[1, 2, 3]], "expected": [2, 1, 3]},
        {"args": [[1, 2, 3, 4, 5, 6, 7]], "expected": [4, 2, 6, 1, 3, 5, 7]},
        {"args": [[1, 2, 3, 4, 5, 6]], "expected": [3, 1, 5, None, 2, 4, 6]},
        {"args": [[-6, -2, 1, 4, 8, 9]], "expected": [1, -6, 8, None, -2, 4, 9]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def build(lo: int, hi: int) -> Optional[TreeNode]:
            if lo > hi:
                return None
            mid = (lo + hi) // 2
            return TreeNode(nums[mid], build(lo, mid - 1), build(mid + 1, hi))

        return build(0, len(nums) - 1)''',
    "explanation": "Divide and conquer on the sorted array. The middle element becomes the root: everything to its left is smaller and goes into the left subtree, everything to its right goes into the right subtree. Both halves differ in size by at most one element, so the two subtrees' heights differ by at most one at every node. The result is a valid BST that is height-balanced.\n\nTime O(n), space O(log n) for the recursion, not counting the output tree.",
    "explanation_zh": "对有序数组做分治。中间的元素作为根节点：它左边的元素都更小，放入左子树；右边的元素都更大，放入右子树。两半的元素个数相差至多 1，所以每个节点两棵子树的高度差都不超过 1，得到的是一棵高度平衡的二叉搜索树。\n\n时间 O(n)，递归空间 O(log n)（不计输出的树）。",
}
