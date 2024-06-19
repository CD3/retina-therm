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


def test_interpolating():
    t = numpy.array(numpy.arange(0, 2 + 0.1, 0.1))
    T = 2 * t

    tp = numpy.array(numpy.arange(0, 2 + 0.01, 0.01))
    Tp = multi_pulse_builder.interpolate_temperature_history(t, T, tp)

    assert t[0] == pytest.approx(0)
    assert tp[0] == pytest.approx(0)
    assert t[-1] == pytest.approx(2)
    assert tp[-1] == pytest.approx(2)

    assert len(Tp) == 201
    assert len(T) == 21
    assert Tp[0] == pytest.approx(0)
    assert Tp[-1] == pytest.approx(4)
    assert Tp[1] == pytest.approx(0.01 * 2)
    assert Tp[2] == pytest.approx(0.02 * 2)


def test_grid_regulizer():
    t = numpy.zeros([5])
    t[0] = 0
    t[1] = 0.1
    t[2] = 0.3
    t[3] = 0.6
    t[4] = 1.0

    tp = multi_pulse_builder.regularize_grid(t)

    assert len(tp) > len(t)
    assert len(tp) == 11

    assert tp[0] == pytest.approx(0)
    assert tp[1] == pytest.approx(0.1)
    assert tp[2] == pytest.approx(0.2)
    assert tp[-1] == pytest.approx(1)
    assert tp[-2] == pytest.approx(0.9)


def test_regularizing_temperature_history():
    t = numpy.array([0, 0.1, 0.3, 1, 2])
    T = 2 * t

    tp = multi_pulse_builder.regularize_grid(t)
    Tp = multi_pulse_builder.interpolate_temperature_history(t, T, tp)

    assert len(Tp) == 21

    assert Tp[0] == pytest.approx(0)
    assert Tp[20] == pytest.approx(4)
    assert Tp[10] == pytest.approx(2)


def test_finding_index_for_time():
    t = numpy.array([0, 1e-6, 2e-6, 3e-6])

    assert multi_pulse_builder.find_index_for_time(t, -2e-6) is None
    assert multi_pulse_builder.find_index_for_time(t, -1e-6) is None
    assert multi_pulse_builder.find_index_for_time(t, 0.0) == 0
    assert multi_pulse_builder.find_index_for_time(t, 1e-6) == 1
    assert multi_pulse_builder.find_index_for_time(t, 2e-6) == 2
    assert multi_pulse_builder.find_index_for_time(t, 3e-6) == 3
    assert multi_pulse_builder.find_index_for_time(t, 4e-6) is None
    assert multi_pulse_builder.find_index_for_time(t, 5e-6) is None

    t = numpy.array([0, 0.5e-6, 1e-6, 1.5e-6, 2e-6, 3e-6])

    assert multi_pulse_builder.find_index_for_time(t, -2e-6) is None
    assert multi_pulse_builder.find_index_for_time(t, -1e-6) is None
    assert multi_pulse_builder.find_index_for_time(t, 0.0) == 0
    assert multi_pulse_builder.find_index_for_time(t, 1e-6) == 2
    assert multi_pulse_builder.find_index_for_time(t, 2e-6) == 4
    assert multi_pulse_builder.find_index_for_time(t, 3e-6) == 5
    assert multi_pulse_builder.find_index_for_time(t, 4e-6) is None
    assert multi_pulse_builder.find_index_for_time(t, 5e-6) is None
