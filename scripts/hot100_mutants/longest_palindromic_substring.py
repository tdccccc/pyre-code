# Mistake: expands only around characters, so even-length palindromes are never found
class Solution:
    def longestPalindrome(self, s):
        best = s[:1]
        for c in range(len(s)):
            lo = hi = c
            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                lo, hi = lo - 1, hi + 1
            if hi - lo - 1 > len(best):
                best = s[lo + 1:hi]
        return best
