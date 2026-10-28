# Mistake: assumes each repeated value sits right next to its twin, so it compares neighbours in pairs
class Solution:
    def singleNumber(self, nums):
        for i in range(0, len(nums) - 1, 2):
            if nums[i] != nums[i + 1]:
                return nums[i]
        return nums[-1]
