# Mistake: Jumps the window start back to an earlier repeat that is already outside the window
class Solution:
    def lengthOfLongestSubstring(self, s):
        last, start, best = {}, 0, 0
        for i, ch in enumerate(s):
            if ch in last:
                start = last[ch] + 1
            last[ch] = i
            best = max(best, i - start + 1)
        return best
