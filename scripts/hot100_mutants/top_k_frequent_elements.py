# Mistake: returns the first k distinct values in order of appearance instead of the k most frequent
from collections import Counter


class Solution:
    def topKFrequent(self, nums, k):
        return list(Counter(nums))[:k]
