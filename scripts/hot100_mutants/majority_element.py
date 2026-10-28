# Mistake: uses a threshold of n // 2 instead of more than n / 2, so a value that only ties half the array can win
class Solution:
    def majorityElement(self, nums):
        counts = {}
        for x in nums:
            counts[x] = counts.get(x, 0) + 1
            if counts[x] >= len(nums) // 2:
                return x
        return None
