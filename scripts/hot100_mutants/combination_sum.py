# Mistake: every candidate is tried at every step (no start index), so [2,3] and [3,2] are both reported
class Solution:
    def combinationSum(self, candidates, target):
        result, path = [], []

        def backtrack(remaining):
            if remaining == 0:
                result.append(path[:])
                return
            for c in candidates:
                if c <= remaining:
                    path.append(c)
                    backtrack(remaining - c)
                    path.pop()

        backtrack(target)
        return result
