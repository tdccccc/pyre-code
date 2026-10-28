# Mistake: reads each digit as its own repeat count, so 10[a] is read as 0[a]
class Solution:
    def decodeString(self, s):
        stack, cur, num = [], "", 0
        for ch in s:
            if ch.isdigit():
                num = int(ch)
            elif ch == "[":
                stack.append((cur, num))
                cur, num = "", 0
            elif ch == "]":
                prev, times = stack.pop()
                cur = prev + cur * times
            else:
                cur += ch
        return cur
