# Mistake: the left subtree gets preorder[1:mid], one node short, so the split is off by one
class Solution:
    def buildTree(self, preorder, inorder):
        if not preorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])
        return root
