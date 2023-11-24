import copy
import itertools

import scipy
from fspathtree import fspathtree


def bisect(f, a, b, tol=1e-8, max_iter=1000):
    lower = f(a)
    upper = f(b)
    sign = (upper - lower) / abs(upper - lower)
    if lower * upper >= 0:
        raise RuntimeError(
            f"Bracket [{a},{b}] does not contain a zero. f(a) and f(b) should have different signs but they were {lower} and {upper}."
        )

    mid = (a + b) / 2
    f_mid = f(mid)
    num_iter = 0
    while num_iter < max_iter and abs(f_mid) > tol:
        num_iter += 1
        a = mid if sign * f_mid < 0 else a
        b = mid if sign * f_mid > 0 else b
        mid = (a + b) / 2
        f_mid = f(mid)

    return (a, b)


def MarcumQFunction(nu, a, b):
    return 1 - scipy.stats.ncx2.cdf(b**2, 2 * nu, a**2)
