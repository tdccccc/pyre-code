# Mistake: sets the merged end to the new interval's end instead of the larger one, so a contained interval shrinks the result
class Solution:
    def merge(self, intervals):
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        return merged
