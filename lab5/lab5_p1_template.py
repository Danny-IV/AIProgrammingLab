class Fraction(object):
    """Class to represent a number as a fraction"""
    def __init__(self, n, d):
        """ Method to construct a Fraction object """
        if type(n) != int or type(d) != int: # Check that n and d are of type int
            raise ValueError('requires type int')
        if d <= 0:
            raise ZeroDivisionError('requires positive integer denominator')
        # If we get here, n and d are ok => initialize Fraction:
        self.num = n
        self.denom = d
        self.reduce()

    ### Your work begins here ...

    def reduce(self):
        """Reduces self to simplest terms. This is done by dividing both
        numerator and denominator by their greatest common divisor (GCD)."""
        pass
    
    def __str__(self):
        """ Returns a string representation of the fraction object (self) """
        return str(self.num) + '/' + str(self.denom)    # Modify this method.

    def __add__(self, other):
        """ Returns new Fraction representing self + other """
        new_num = self.num * other.denom + other.num * self.denom
        new_denom = self.denom * other.denom
        return Fraction(new_num, new_denom)     # Modify this method.

    def __mul__(self, other):
        """ Returns new Fraction representing self * other """
        new_num = self.num * other.num
        new_denom = self.denom * other.denom
        return Fraction(new_num, new_denom)     # Modify this method.

    def __sub__(self, other):
        pass
    
    def __truediv__(self, other):
        pass
    
    def __lt__(self, other):
        pass
    
    def __le__(self, other):
        pass
    
    def __gt__(self, other):
        pass
    
    def __ge__(self, other):
       pass
    
    def __eq__(self, other):
        pass
    
    ### Your work ends here ...
    # ----- You don't have to modify below
    def __gcd(self, _a, _b):
        """Get GCD of _a and _b"""
        a, b = abs(_a), abs(_b)
        if b > a: a, b = b, a
        while b != 0:
            [a, b] = [b, a%b]
        if a == 0: a = 1
        return a

    def __neg__(self):
        """Negation"""
        # Returns -self
        return Fraction(-self.num, self.denom)

    __radd__ = __add__
    __rmul__ = __mul__
    
    def __rsub__(self, other):
        """Right subtraction"""
        # Handles int - Fraction
        if isinstance(other, int):
            return Fraction(other, 1).__sub__(self)
        elif isinstance(other, Fraction):
            return other.__sub__(self)
        else:
            raise ValueError('requires type int or Fraction')
    
    def __rtruediv__(self, other):
        """Right division"""
        # Handles int / Fraction
        if isinstance(other, int):
            return Fraction(other, 1).__truediv__(self)
        elif isinstance(other, Fraction):
            return other.__truediv__(self)
        else:
            raise ValueError('requires type int or Fraction')
    
    def __pow__(self, other):
        """Power; Fraction ** int"""
        if isinstance(other, int):
            if other == 0:
                return Fraction(1, 1)
            else:
                ret = self
                for _ in range(abs(other) - 1):
                    ret = ret.__mul__(self)
                if other < 0:
                    ret = ret.__rtruediv__(1)
            return ret
        else:
            raise ValueError('requires type int')

