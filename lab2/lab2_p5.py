import math


def myPiCalculator(tol):
    """calculating pi by iterating math"""

    # initial value of pi and term
    pi = 2
    term = math.sqrt(2)

    # iterating until diff is under tolerance
    while True:
        pi_new = pi * 2 / term
        if abs(pi - pi_new) < tol:
            pi = pi_new
            break
        else:
            pi = pi_new
            term = math.sqrt(2 + term)
    return pi
