# Mistake: checks only that each distinct letter of t is present, ignoring how many times it is needed
from collections import Counter


class Solution:
    def minWindow(self, s, t):
        need = set(t)
        have = Counter()
        best = None
        left = 0
        for right, ch in enumerate(s):
            have[ch] += 1
            while all(have[c] > 0 for c in need):
                if best is None or right - left + 1 < len(best):
                    best = s[left:right + 1]
                have[s[left]] -= 1
                left += 1
        return best or ""
