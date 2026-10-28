# Mistake: compares the set of letters in each window with p's set, ignoring how many times each letter occurs
class Solution:
    def findAnagrams(self, s, p):
        m = len(p)
        target = set(p)
        return [i for i in range(len(s) - m + 1) if set(s[i:i + m]) == target]
