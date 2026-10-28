# Mistake: divides the total product by each element, which fails as soon as a zero is present
class Solution:
    def productExceptSelf(self, nums):
        total = 1
        for x in nums:
            total *= x
        return [total // x for x in nums]
