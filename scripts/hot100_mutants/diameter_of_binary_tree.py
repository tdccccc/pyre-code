# Mistake: only measures the route through the root, never a route inside one subtree
class Solution:
    def diameterOfBinaryTree(self, root):
        def depth(n):
            return 0 if n is None else 1 + max(depth(n.left), depth(n.right))

        if root is None:
            return 0
        return depth(root.left) + depth(root.right)
