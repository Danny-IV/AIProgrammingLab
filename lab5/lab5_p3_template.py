import math
try:
    from lab5_p1 import Fraction
except ImportError:
    pass

class Polynomial:
    def __init__(self, coefficients: list):
        """Initializes the polynomial with a list of integer or Fraction coefficients."""
        self.coefficients = coefficients
        ### Solution version below
        # self.coefficients = []
        # for c in coefficients:
        #     if isinstance(c, int):
        #         self.coefficients.append(Fraction(c, 1))
        #     elif isinstance(c, Fraction):
        #         self.coefficients.append(c)

    def __add__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def derivative(self):
        pass

    def integral(self, C=0):
        pass

    def evaluate(self, x: int):
        """Return f(x) where f is self polynomial"""
        result = 0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (x ** i)
        return result

    def __str__(self):
        """String representation of polynomial equation"""
        terms = []
        for i, coeff in enumerate(self.coefficients):
            if coeff != 0:
                if i == 0:
                    terms.append(f"{coeff}")
                elif i == 1:
                    terms.append(f"{coeff}*x")
                else:
                    terms.append(f"{coeff}*x^{i}")
        return " + ".join(terms).replace('+ -', '- ') if terms else "0"

