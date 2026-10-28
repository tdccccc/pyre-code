# Mistake: records the first node reached at each depth (the left view), not the rightmost one
class Solution:
    def rightSideView(self, root):
        view = []

        def dfs(node, depth):
            if not node:
                return
            if depth == len(view):
                view.append(node.val)
            dfs(node.left, depth + 1)
            dfs(node.right, depth + 1)

        dfs(root, 0)
        return view
