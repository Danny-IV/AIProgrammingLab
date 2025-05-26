import math

try:
    from lab5_p1 import Fraction
except ImportError:
    pass


class Polynomial:
    """Represents polynomial, with fraction"""

    def __init__(self, coefficients: list = None):
        """Initializes the polynomial with a list of integer or Fraction coefficients."""
        # set default coefficient
        if coefficients is None:
            coefficients = []

        self.coefficients = []
        for c in coefficients:
            if isinstance(c, int):
                self.coefficients.append(Fraction(c, 1))
            elif isinstance(c, Fraction):
                self.coefficients.append(c)

    def __add__(self, other):
        """Returns addition result of polynomial"""
        length = max(len(self.coefficients), len(other.coefficients))
        # Initialize new polynomial
        result = Polynomial()
        for i in range(length):
            a = self.coefficients[i] if i < len(self.coefficients) else 0
            b = other.coefficients[i] if i < len(other.coefficients) else 0
            result.coefficients.append(a + b)
        return result

    def __sub__(self, other):
        """Returns subtraction result of polynomial"""
        length = max(len(self.coefficients), len(other.coefficients))
        # Initialize new polynomial
        result = Polynomial()
        for i in range(length):
            a = self.coefficients[i] if i < len(self.coefficients) else 0
            b = other.coefficients[i] if i < len(other.coefficients) else 0
            result.coefficients.append(a - b)
        return result

    def __mul__(self, other):
        """Returns multiplication result of polynomial"""
        length = len(self.coefficients) * len(other.coefficients)
        result = [0] * length
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                result[i + j] += a * b
        return Polynomial(result)

    def derivative(self):
        """Returns derivative result of polynomial"""
        result = Polynomial()
        for i in range(1, len(self.coefficients)):
            result.coefficients.append(self.coefficients[i] * i)
        return result

    def integral(self, C=0):
        """Returns integral result of polynomial"""
        result = Polynomial([C])
        for i, c in enumerate(self.coefficients):
            result.coefficients.append(c / (i + 1))
        return result

    def evaluate(self, x: int):
        """Return f(x) where f is self polynomial"""
        result = 0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (x**i)
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
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"
