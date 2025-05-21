import math

class Point:
    """2D Euclidean coordinate"""
    def __init__(self, x, y):
        """Initialization"""
        self.x = round(x, 2)
        self.y = round(y, 2)

    def __str__(self):
        """"""
        return f"({self.x:.2f}, {self.y:.2f})"
    
    def __add__(self, other):
        """P1 + P2"""
        return Point(self.x + other.x, self.y + other.y)

    def __neg__(self):
        """Negation (-P1)"""
        return Point(-self.x, -self.y)

    def __sub__(self, other):
        """P1 - P2"""
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):
        """Note: other is int or float"""
        if isinstance(other, float) or isinstance(other, int):
            return Point(other * self.x, other * self.y)
        else:
            raise ValueError("other should be type float or int")
    
    def __rmul__(self, other):
        """int * P1"""
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
        # Your code here ...
        # return Point(x, y)
        return
    
    def rotate(self, reference_point, angle):
        # Your code here ...
        # return Point(x, y)
        return


