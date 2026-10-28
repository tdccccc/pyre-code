# Mistake: Swaps zeroes to the back, which scrambles the order of the non-zeroes
class Solution:
    def moveZeroes(self, nums):
        i, j = 0, len(nums) - 1
        while i < j:
            if nums[i] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                j -= 1
            else:
                i += 1
