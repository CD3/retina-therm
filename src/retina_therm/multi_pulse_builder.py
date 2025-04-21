import copy
from typing import Optional

import numpy
import scipy

from .signals import Signal


def is_uniform_spaced(x: numpy.array, tol: float = 1e-10):
    dx = x[1] - x[0]
    for i in range(len(x) - 1):
        if (x[i + 1] - x[i]) - dx > tol:
            return False
    return True


def is_resolution(x: numpy.array, res: float, tol: float = 1e-10):
    if is_uniform_spaced(x, tol):
        return abs((x[1] - x[0]) - res) < tol
    return False


def regularize_grid(t, dt=None):
    if dt is None:
        dt = t[1] - t[0]
    tmax = t[-1]
    tmin = t[0]
    N = int((tmax - tmin) / dt) + 1
    tp = numpy.zeros([N])
    for i in range(N):
        tp[i] = tmin + i * dt

    return tp


def interpolate_temperature_history(t, T, tp):
    """Interpolate a temperature history to a differt set of times."""
    Tp = numpy.zeros([len(tp)])

    interp = scipy.interpolate.PchipInterpolator(t, T)
    for i in range(len(tp)):
        if tp[i] < t[0] or tp[i] > t[-1]:
            continue
        Tp[i] = interp(tp[i])

    return Tp


def find_index_for_time(ts: numpy.array, t: float, tol: float = 1e-6) -> int:
    """
    Search array of times for a given time t and return its index. Matche does not have to be exact,
    anything within tol (default to 1 us) will be considered a match.
    """
    if t < ts[0]:
        return None
    if t > ts[-1]:
        return None
    N = len(ts)
    for i in range(N):
        diff = abs(ts[i] - t)
        if diff < tol:
            # check if there are any other times that are closer to the value we are looking for
            j = i + 1
            while j < N - 1:
                diff2 = abs(ts[j] - t)
                if diff2 < diff:
                    diff = diff2
                    j += 1
                else:
                    break
            return j - 1
    return None


class MultiPulseBuilder:
    def __init__(self):
        self.T0 = 0
        self.dT = None
        self.t = None

        self.arrival_times = []
        self.scales = []

        self.progress = Signal()
        self.status = Signal()

    def set_baseline_temperature(self, val: float) -> None:
        self.T0 = val

    def set_temperature_history(self, t: numpy.array, T: numpy.array):
        assert len(t) > 0
        assert len(T) > 0
        assert len(t) == len(T)
        self.T0 = T[0]

        if not is_uniform_spaced(t):
            raise RuntimeError(
                "Currently only support uniform spacing of the temperature history."
            )

        self.t = copy.copy(t)
        self.dT = T - self.T0

    def add_contribution(self, t: float, scale: float) -> None:
        """Add a contribution to the thermal profile."""
        self.arrival_times.append(t)
        self.scales.append(scale)

    def clear_contributions(self) -> None:
        self.arrival_times = []
        self.scales = []

    def build(self) -> numpy.array:
        t = self.t

        if not is_uniform_spaced(t):
            raise RuntimeError(
                "Currently only support uniform spacing of the temperature history."
            )

        T = numpy.zeros([len(t)])

        N = len(self.arrival_times)
        if len(t) < 2:
            raise RuntimeError("Temperature history only contains 1 point.")

        dt = t[1] - t[0]
        for i in range(N):
            if self.arrival_times[i] > t[-1]:
                continue

            offset = find_index_for_time(t, self.arrival_times[i], dt)
            if offset is None:
                raise RuntimeError(
                    f"Could not find a time point in the single-pulse exposure that was close enough to {self.arrival_times[i]}. This could mean your input data is too low resolution for the pulse duration / pulse period. t = {t}"
                )

            if offset != 0:
                T[offset:] += self.scales[i] * self.dT[:-offset]
            else:
                T += self.scales[i] * self.dT
            self.progress.emit(i, N)

        T += self.T0

        return T
