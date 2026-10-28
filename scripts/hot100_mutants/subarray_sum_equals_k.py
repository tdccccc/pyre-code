# Mistake: sliding window that shrinks while the sum exceeds k, which only works when every number is non-negative
class Solution:
    def subarraySum(self, nums, k):
        count = 0
        left = total = 0
        for right, x in enumerate(nums):
            total += x
            while total > k and left <= right:
                total -= nums[left]
                left += 1
            if total == k:
                count += 1
        return count
