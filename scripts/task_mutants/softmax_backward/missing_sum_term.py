# Mistake: omits the -sum(dy * y) correction, returning y * dy (only the diagonal of the Jacobian)
def softmax_backward(y, dy):
    return y * dy  # BUG: missing the rank-one term y * sum(dy * y)
