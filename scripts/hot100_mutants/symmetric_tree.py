# Mistake: compares the in-order values with their reverse, so shape is ignored
class Solution:
    def isSymmetric(self, root):
        vals = []

        def inorder(node):
            if node:
                inorder(node.left)
                vals.append(node.val)
                inorder(node.right)

        inorder(root)
        return vals == vals[::-1]
