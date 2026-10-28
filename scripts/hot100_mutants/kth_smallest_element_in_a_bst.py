# Mistake: walks the tree in pre-order (node before its subtrees) and returns the k-th node visited
class Solution:
    def kthSmallest(self, root, k):
        order, stack = [], [root]
        while stack:
            node = stack.pop()
            order.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return order[k - 1]
