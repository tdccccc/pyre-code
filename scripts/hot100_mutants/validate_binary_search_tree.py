# Mistake: only compares each node with its direct children, never with its ancestors' bounds
class Solution:
    def isValidBST(self, root):
        if root is None:
            return True
        if root.left and root.left.val >= root.val:
            return False
        if root.right and root.right.val <= root.val:
            return False
        return self.isValidBST(root.left) and self.isValidBST(root.right)
