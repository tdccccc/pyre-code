# Mistake: Stores x before looking up target - x, so x can pair with itself
class Solution:
    def twoSum(self, nums, target):
        s = {}
        for i, x in enumerate(nums):
            s[x] = i
            if target - x in s:
                return [s[target - x], i]
