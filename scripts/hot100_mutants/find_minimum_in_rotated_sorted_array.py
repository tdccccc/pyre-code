# Mistake: compares each middle value with nums[0], so an unrotated array returns its last element
class Solution:
    def findMin(self, nums):
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] >= nums[0]:
                lo = mid + 1
            else:
                hi = mid
        return nums[lo]
