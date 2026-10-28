# Mistake: updates dy in place to save memory, which corrupts the caller's upstream gradient
def softmax_backward(y, dy):
    s = (dy * y).sum(dim=-1, keepdim=True)
    dy.sub_(s)  # BUG: mutates the input
    return y * dy
