# Mistake: swaps the root's children but forgets to recurse into the subtrees
class Solution:
    def invertTree(self, root):
        if root is not None:
            root.left, root.right = root.right, root.left
        return root
