import math


class Point:
    """2D Euclidean coordinate"""

    def __init__(self, x, y):
        """Initializatize 2d coordinate"""
        self.x = round(x, 2)
        self.y = round(y, 2)

    def __str__(self):
        """String representation of 2d coordinate"""
        return f"({self.x:.2f}, {self.y:.2f})"

    def __add__(self, other):
        """P1 + P2, addition of 2d coordinate"""
        return Point(self.x + other.x, self.y + other.y)

    def __neg__(self):
        """-P1, negation of 2d coordinate"""
        return Point(-self.x, -self.y)

    def __sub__(self, other):
        """P1 - P2, substraction of 2d coordinate"""
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        """P1 * P2, multiplication of 2d coordinate by int or float
        Note: other is int or float"""
        if isinstance(other, float) or isinstance(other, int):
            return Point(other * self.x, other * self.y)
        else:
            raise ValueError("other should be type float or int")

    def __rmul__(self, other):
        """int * P1, right multiplication"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Note: other is int or float"""
        if isinstance(other, float) or isinstance(other, int):
            if other != 0:
                return Point(self.x / other, self.y / other)
            else:
                raise ZeroDivisionError("other should be non-zero")
        else:
            raise ValueError("other should be type float or int")

    ### Your work begins here ...

    def symmetric(self, reference_point):
        """Returns symmetry point by reference point"""
        # calculate symmetry by reference point
        x = self.x + 2 * (reference_point.x - self.x)
        y = self.y + 2 * (reference_point.y - self.y)
        return Point(x, y)

    def rotate(self, reference_point, angle):
        """Returns rotated point by reference point and angle"""
        # calculate rotation by reference point
        deltaX = self.x - reference_point.x
        deltaY = self.y - reference_point.y
        x = deltaX * math.cos(angle) - deltaY * math.sin(angle)
        y = deltaX * math.sin(angle) + deltaY * math.cos(angle)
        x += reference_point.x
        y += reference_point.y
        return Point(x, y)
