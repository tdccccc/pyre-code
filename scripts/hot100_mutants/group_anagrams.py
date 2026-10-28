# Mistake: Groups by the set of letters, ignoring how often each occurs
class Solution:
    def groupAnagrams(self, strs):
        g = {}
        for s in strs:
            g.setdefault(frozenset(s), []).append(s)
        return list(g.values())
