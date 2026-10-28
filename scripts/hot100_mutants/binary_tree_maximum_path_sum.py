# Mistake: a path can only go up through a node, never bend at it (best = value + one branch)
class Solution:
    def maxPathSum(self, root):
        best = float("-inf")

        def gain(node):
            nonlocal best
            if not node:
                return 0
            left = max(gain(node.left), 0)
            right = max(gain(node.right), 0)
            best = max(best, node.val + max(left, right))
            return node.val + max(left, right)

        gain(root)
        return best
