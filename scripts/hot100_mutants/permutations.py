# Mistake: appends the shared path list instead of a copy, so every answer ends up as the same empty list
class Solution:
    def permute(self, nums):
        result, path = [], []

        def backtrack():
            if len(path) == len(nums):
                result.append(path)
                return
            for x in nums:
                if x not in path:
                    path.append(x)
                    backtrack()
                    path.pop()

        backtrack()
        return result
