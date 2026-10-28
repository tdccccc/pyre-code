# Mistake: takes the minimum child depth instead of the maximum (that is the minimum depth, LeetCode 111)
class Solution:
    def maxDepth(self, root):
        if root is None:
            return 0
        return 1 + min(self.maxDepth(root.left), self.maxDepth(root.right))
