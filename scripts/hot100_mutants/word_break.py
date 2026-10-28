# Mistake: greedily takes the longest dictionary word at each position and never backtracks
class Solution:
    def wordBreak(self, s, wordDict):
        words = set(wordDict)
        i = 0
        while i < len(s):
            for j in range(len(s), i, -1):
                if s[i:j] in words:
                    i = j
                    break
            else:
                return False
        return True
