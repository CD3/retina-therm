import math

from mpmath import mp

mp.dsp = 1000

import numpy
import pytest
import scipy

from retina_therm import greens_functions
from retina_therm.units import *


def test_cw_retina_exposure():
    exp = greens_functions.CWRetinaLaserExposure(
        {
            "laser": {
                "E0": "1 W/cm^2",
                "one_over_e_radius": "1 cm",
            },
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "0.0 um",
                },
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "10 um",
                },
            ],
            "simulation": {
                "with_units": False,
                "use_multi_precision": False,
                "use_approximate": True,
            },
        }
    )

    T = exp.temperature_rise(0, 0, numpy.arange(0, 100e-6, 1e-6), method="trap")
    # need to find some actual values to compare against, these are just values
    # that were produced at one point but change when any adjustments are made intenrally...
    # assert T[-1] == pytest.approx(9.992091714864e-6)

    exp2 = greens_functions.CWRetinaLaserExposure(
        {
            "laser": {"E0": "1 W/cm^2", "one_over_e_radius": "1 cm", "duration": "50 us"},
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "0.0 um",
                },
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "10 um",
                },
            ],
            "simulation": {
                "with_units": False,
                "use_multi_precision": False,
                "use_approximate": True,
            },
        }
    )

    T1 = exp.temperature_rise(0, 0, numpy.arange(0, 100e-6, 1e-6), method="trap")[-1]
    T2 = exp.temperature_rise(0, 0, numpy.arange(0, 50e-6, 1e-6), method="trap")[-1]
    T3 = exp2.temperature_rise(0, 0, numpy.arange(0, 100e-6, 1e-6), method="trap")
    # see not above..
    # assert T3[-1] == pytest.approx(T1 - T2)


def test_pulsed_retina_exposure():
    exp = greens_functions.PulsedRetinaLaserExposure(
        {
            "laser": {"E0": "1 W/cm^2", "one_over_e_radius": "1 cm", "pulse_duration": "1 year"},
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "0.0 um",
                },
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "10 um",
                },
            ],
            "simulation": {
                "with_units": False,
                "use_multi_precision": False,
                "use_approximate": True,
            },
        }
    )

    T = exp.temperature_rise(0, 0, numpy.arange(0, 100e-6, 1e-6), method="trap")
    # need to find some actual values to compare against, these are just values
    # that were produced at one point but change when any adjustments are made intenrally...
    # assert T[-1] == pytest.approx(9.992091714864e-6)

    exp2 = greens_functions.PulsedRetinaLaserExposure(
        {
            "laser": {"E0": "1 W/cm^2", "one_over_e_radius": "1 cm", "pulse_duration": "50 us"},
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "0.0 um",
                },
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "10 um",
                },
            ],
            "simulation": {
                "with_units": False,
                "use_multi_precision": False,
                "use_approximate": True,
            },
        }
    )

    T1 = exp.temperature_rise(0, 0, numpy.arange(0, 100e-6, 1e-6), method="trap")[-1]
    T2 = exp.temperature_rise(0, 0, numpy.arange(0, 50e-6, 1e-6), method="trap")[-1]
    T3 = exp2.temperature_rise(0, 0, numpy.arange(0, 100e-6, 1e-6), method="trap")
    # see not above..
    # assert T3[-1] == pytest.approx(T1 - T2)


def test_gf_integrators():
    G = greens_functions.MultiLayerGreensFunction(
        {
            "laser": {"E0": "1 W/cm^2", "one_over_e_radius": "1 cm", "duration": "50 us"},
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "0.0 um",
                },
                {
                    "mua": "1 1/cm",
                    "d": "10 um",
                    "z0": "10 um",
                },
            ],
            "simulation": {
                "with_units": False,
                "use_multi_precision": False,
                "use_approximate": True,
            },
        }
    )
    integrator = greens_functions.GreensFunctionTrapezoidIntegrator(G)
    T = integrator.temperature_rise(
        0, 0, numpy.arange(0, 0.001, 0.0001), {"duration": "0.001 s"}
    )
    assert len(T) == 10
