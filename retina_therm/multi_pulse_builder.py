import copy
from typing import Optional

import numpy


class MultiPulseBuilder:
    def __init__(self):
        self.T0 = 0
        self.dT = None
        self.t = None

        self.arrival_times = []
        self.scales = []

    def set_baseline_temperature(self, val: float) -> None:
        self.T0 = val

    def set_temperature_history(self, t: numpy.array, T: numpy.array):
        assert len(t) > 0
        assert len(T) > 0
        assert len(t) == len(T)
        self.T0 = T[0]

        if not self.is_uniform_spaced(t):
            raise RuntimeError(
                "Currently only support uniform spacing of the temperature history."
            )

        self.t = copy.copy(t)
        self.dT = T - self.T0

    def find_index_for_time(self, t: float, tol: float = 1e-10) -> int:
        for i in range(len(self.t)):
            if abs(self.t[i] - t) < tol:
                return i

    def add_contribution(self, t: float, scale: float) -> None:
        """Add a contribution to the thermal profile."""
        self.arrival_times.append(t)
        self.scales.append(scale)

    def clear_contributions(self) -> None:
        self.arrival_times = []
        self.scales = []

    def is_uniform_spaced(self, x: numpy.array, tol: float = 1e-10):
        dx = x[1] - x[0]
        for i in range(len(x) - 1):
            if (x[i + 1] - x[i]) - dx > tol:
                return False
        return True

    def build(self) -> numpy.array:
        t = self.t

        if not self.is_uniform_spaced(t):
            raise RuntimeError(
                "Currently only support uniform spacing of the temperature history."
            )

        T = numpy.zeros([len(t)])

        for i in range(len(self.arrival_times)):
            if self.arrival_times[i] > t[-1]:
                continue

            offset = self.find_index_for_time(self.arrival_times[i])

            if offset != 0:
                T[offset:] += self.scales[i] * self.dT[:-offset]
            else:
                T += self.dT

        T += self.T0

        return T
