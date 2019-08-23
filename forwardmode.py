
import numpy as np

from typing import Optional


class Dual(object):
    def __init__(self, real: float, dual: Optional[float]) -> None:
        self.real = float(real)
        self.dual = float(dual) if dual else 0.

    @classmethod
    def to_dual(cls, other):
        try:
            return cls(other)
        except TypeError:
            return other

    def __add__(self, other):
        other = Dual.to_dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        other = Dual.to_dual(other)
        return Dual(self.real - other.real, self.dual - other.dual)

    def __mul__(self, other):
        other = Dual.to_dual(other)
        return Dual(self.real * other.real, self.real * other.dual + self.dual * other.real)

    def __truediv__(self, other):
        other = Dual.to_dual(other)
        try:
            return Dual(self.real / other.real,
                        (self.dual * other.real - self.real * other.dual) / (other.dual ** 2))
        except ZeroDivisionError:
            raise ZeroDivisionError("Divisor in dual division must have real part != 0")

    def __pow__(self, power, modulo=None):
        if modulo:
            raise NotImplementedError
        power = Dual.to_dual(power)
        return Dual.exp(power * Dual.log(self))

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def exp(self):
        real_exp = np.exp(self.real)
        return Dual(real_exp, real_exp * self.dual)

    def log(self):
        try:
            return Dual(np.log(self.real), self.dual / self.real)
        except ZeroDivisionError:
            raise ZeroDivisionError("log must have dual real part != 0")

    def sqrt(self):
        real_sqrt = np.sqrt(self.real)
        try:
            return Dual(real_sqrt, 0.5 * self.dual / real_sqrt)
        except ZeroDivisionError:
            raise ZeroDivisionError("sqrt must have dual real part != 0")

    def sin(self):
        return Dual(np.sin(self.real), self.dual * np.cos(self.real))

    def cos(self):
        return Dual(np.cos(self.real), -self.dual * np.sin(self.real))

    def __repr__(self):
        return "Dual({}, {})".format(self.real, self.dual)


if __name__ == '__main__':

    def f(x, y): return x * y + np.sin(x)
    def f_prime_x(x, y): return y + np.cos(x)
    def f_prime_y(x, y): return x

    x = 4.56
    y = 1.23

    print("with reals!")
    print('f({}, {}) = {}'.format(x, y, f(x, y)))
    print('f_prime_x({}, {}) = {}'.format(x, y, f_prime_x(x, y)))
    print('f_prime_y({}, {}) = {}'.format(x, y, f_prime_y(x, y)))
    print()

    x_dual = Dual(x, 0)
    y_dual = Dual(y, 0)
    x_dual_prime = Dual(x, 1)
    y_dual_prime = Dual(y, 1)

    print("with duals!")
    print('f({}, {}) = {}'.format(x_dual, y_dual, f(x_dual, y_dual)))
    print('f_prime_x({}, {}) = {}'.format(x_dual_prime, y_dual, f(x_dual_prime, y_dual)))
    print('f_prime_y({}, {}) = {}'.format(x_dual, y_dual_prime, f(x_dual, y_dual_prime)))
