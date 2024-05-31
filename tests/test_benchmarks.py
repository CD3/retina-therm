import math

import numpy
import pytest
import scipy



def test_numpy_exp(benchmark):
    benchmark(numpy.exp, 0.1)


def test_math_exp(benchmark):
    benchmark(math.exp, 0.1)


def test_scipy_erf(benchmark):
    benchmark(scipy.special.erf, 0.1)


def test_math_erf(benchmark):
    benchmark(math.erf, 0.1)

def test_python_marcum_q(benchmark):
    benchmark(math.erf, 0.1)
