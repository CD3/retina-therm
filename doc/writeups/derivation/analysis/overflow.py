import math

import mpmath
import numpy
import pint
import scipy

ureg = pint.UnitRegistry(system="cgs")
Q_ = ureg.Quantity

rho = Q_(1, "g/cm^3")
c = Q_(1, "cal/g/K")
k = Q_(0.00628, "W/cm/K")

alpha = (k / rho / c).to_base_units()

print("vvvvvvvvvvvvvvvvvv")
mua = Q_(1000, "1/cm")
tp_max = (700 / alpha / mua**2).to_base_units()
print("tp_max:", tp_max)
tp_max = (700**2 / alpha / mua**2).to_base_units()
print("tp_max 2:", tp_max)

min_arg = None
max_arg = None
for x in numpy.logspace(1, 10, 10):
    try:
        v = math.exp(x)
    except OverflowError:
        max_arg = x
        min_arg = 10 ** (numpy.log10(x) - 1)
        break

while 2 * (max_arg - min_arg) / (max_arg + min_arg) > 0.001:
    x = (max_arg + min_arg) / 2
    try:
        v = math.exp(x)
        min_arg = x
    except OverflowError:
        max_arg = x

print("math:", min_arg, "-", max_arg)


min_arg = None
max_arg = None
for x in numpy.logspace(1, 10, 10):
    v = numpy.exp(x)
    if v == float("inf"):
        max_arg = x
        min_arg = 10 ** (numpy.log10(x) - 1)
        break

while 2 * (max_arg - min_arg) / (max_arg + min_arg) > 0.001:
    x = (max_arg + min_arg) / 2
    v = numpy.exp(x)
    if v == float("inf"):
        max_arg = x
    else:
        min_arg = x

print("numpy:", min_arg, "-", max_arg)


def erf_approx_1(x):
    return 1 - numpy.exp(-(x**2)) / x / numpy.sqrt(numpy.pi)


def erf_approx_2(x):
    return 1 - (numpy.exp(-(x**2)) / x / numpy.sqrt(numpy.pi)) * (1 - 1 / 2 / x**2)


def percent_error(approx, actual):
    return numpy.abs((approx - actual) / actual)


for error in [0.1, 0.01, 0.001, 0.0001]:
    x1 = scipy.optimize.brentq(
        lambda x: percent_error(erf_approx_1(x), math.erf(x)) - error, 1e-10, 10
    )
    x2 = scipy.optimize.brentq(
        lambda x: percent_error(erf_approx_2(x), math.erf(x)) - error, 1e-10, 10
    )

    print(error, x1, x2)


tp_min = (1.86 / mua**2 / alpha).to("s")


print("tp_min:", tp_min)
