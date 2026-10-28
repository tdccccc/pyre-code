# Mistake: starts from an empty result list, so the empty subset is never produced
class Solution:
    def subsets(self, nums):
        result = []
        for x in nums:
            result += [[x]] + [s + [x] for s in result]
        return result
