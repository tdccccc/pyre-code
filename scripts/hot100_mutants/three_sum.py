# Mistake: never skips repeated values, so the same triplet is reported several times
class Solution:
    def threeSum(self, nums):
        nums = sorted(nums)
        res = []
        for i in range(len(nums)):
            lo, hi = i + 1, len(nums) - 1
            while lo < hi:
                s = nums[i] + nums[lo] + nums[hi]
                if s == 0:
                    res.append([nums[i], nums[lo], nums[hi]])
                    lo += 1
                    hi -= 1
                elif s < 0:
                    lo += 1
                else:
                    hi -= 1
        return res
