import numpy
import pytest

from retina_therm import multi_pulse_builder


def test_construction():
    mp_builder = multi_pulse_builder.MultiPulseBuilder()
    t = numpy.array(numpy.arange(0, 2 + 0.1, 0.1))
    T = 2 * t

    assert len(t) == 21
    assert T[-1] == pytest.approx(4)

    mp_builder.set_temperature_history(t, T)
    mp_builder.set_baseline_temperature(10)

    new_T = mp_builder.build()

    assert new_T[0] == 10
    assert new_T[-1] == 10

    mp_builder.add_contribution(0, 1)
    new_T = mp_builder.build()

    assert new_T[0] == 10
    assert new_T[10] == 10 + 2
    assert new_T[-1] == 10 + 4

    mp_builder.add_contribution(1, -1)
    new_T = mp_builder.build()

    assert new_T[0] == 10
    assert new_T[10] == 10 + 2
    assert new_T[-1] == 10 + 2
