# Mistake: no base case for p and q, so a node is never reported as its own ancestor
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root:
            return None
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left or right
