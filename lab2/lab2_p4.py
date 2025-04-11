def evalPolynomial(x, L):
    """evaluate polynomial, n_0 + n_1*x**1 + n_2*x**2 + ..."""
    result = 0
    # calculate all polynomial by list
    for i in range(len(L)):
        # add n_i * x**i  to result
        result += L[i] * x**i
    return result
