# Mistake: only checks that each bracket type occurs as often as its partner, ignoring order and nesting
class Solution:
    def isValid(self, s):
        return all(s.count(o) == s.count(c) for o, c in ("()", "[]", "{}"))
