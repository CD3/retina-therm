import math

from mpmath import mp

mp.dsp = 1000

import numpy
import pytest
import scipy

from retina_therm import greens_functions
from retina_therm.units import *


def test_large_beam_call_function():
    G = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "1 cm",
            "z0": "0 cm",
            "with_units": True,
            "use_approximate": False,
        }
    )

    assert G(Q_(0, "cm"), Q_(0, "cm"), Q_(0, "s")).to("K/s") == Q_(1 / 2, "K/s")
    assert G(Q_(1, "cm"), Q_(0, "cm"), Q_(0, "s")).to("K/s").magnitude == pytest.approx(
        Q_(1 / 2, "K/s") * math.exp(-1)
    )
    assert G(Q_(1, "cm"), Q_(0, "cm"), Q_(1, "s")).to("K/s").magnitude == pytest.approx(
        Q_(1 / 2, "K/s").magnitude
        * math.exp(-1)
        * math.exp(1)
        * (scipy.special.erf(1) - scipy.special.erf(-1 / math.sqrt(4) + 1))
    )

    G = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "1 cm",
            "z0": "0 cm",
            "with_units": False,
            "use_approximate": False,
        }
    )
    assert G(1, 0, 1) == pytest.approx(
        Q_(1 / 2, "K/s").magnitude
        * math.exp(-1)
        * math.exp(1)
        * (scipy.special.erf(1) - scipy.special.erf(-1 / math.sqrt(4) + 1))
    )


def test_axial_part_retina():
    G = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "120 1/cm",
            "k": "0.00628 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "4.1868 J/g/K",
            "E0": "1 W/cm^2",
            "d": "12 um",
            "z0": "0 cm",
            "with_units": False,
        }
    )


def test_axial_part_retina_long_time_approx():
    G_exact = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1200 1/cm",
            "k": "0.00628 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "4.1868 J/g/K",
            "E0": "1 W/cm^2",
            "d": "12 um",
            "z0": "0 cm",
            "with_units": False,
            "use_multi_precision": True,
            "use_approximate": False,
        }
    )
    G_approx = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1200 1/cm",
            "k": "0.00628 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "4.1868 J/g/K",
            "E0": "1 W/cm^2",
            "d": "12 um",
            "z0": "0 cm",
            "with_units": False,
            "use_multi_precision": False,
            "use_approximate": True,
        }
    )
    G_approx_with_units = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1200 1/cm",
            "k": "0.00628 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "4.1868 J/g/K",
            "E0": "1 W/cm^2",
            "d": "12 um",
            "z0": "0 cm",
            "with_units": True,
            "use_multi_precision": False,
            "use_approximate": True,
        }
    )

    assert G_exact(0, 0, 0.0) == pytest.approx(G_approx(0, 0, 0.0))
    assert G_exact(0, 0, 0.0) == pytest.approx(
        G_approx_with_units(Q_(0, "cm"), Q_(0, "cm"), Q_(0.0, "s")).magnitude
    )
    assert type(G_exact(0, 0, 0.0)) == mp.mpf
    assert type(G_approx(0, 0, 0.0)) == numpy.float64

    assert G_exact(0, 0, 0.01) == pytest.approx(G_approx(0, 0, 0.01), rel=0.01)
    assert G_exact(0, 0, 0.01) == pytest.approx(
        G_approx_with_units(Q_(0, "cm"), Q_(0, "cm"), Q_(0.01, "s")).magnitude, rel=0.01
    )
    assert type(G_exact(0, 0, 0.01)) == mp.mpf
    assert type(G_approx(0, 0, 0.01)) == numpy.float64

    with mp.workdps(200):
        assert G_exact(0, 0, 0.2) == pytest.approx(G_approx(0, 0, 0.2), rel=0.01)
        assert G_exact(0, 0, 0.2) == pytest.approx(
            G_approx_with_units(Q_(0, "cm"), Q_(0, "cm"), Q_(0.2, "s")).magnitude,
            rel=0.01,
        )
        assert type(G_exact(0, 0, 0.2)) == mp.mpf
        assert type(G_approx(0, 0, 0.2)) == numpy.float64

    with mp.workdps(2000):
        assert G_exact(0, 0, 2) == pytest.approx(G_approx(0, 0, 2), rel=0.01)
        assert G_exact(0, 0, 2) == pytest.approx(
            G_approx_with_units(Q_(0, "cm"), Q_(0, "cm"), Q_(2, "s")).magnitude,
            rel=0.01,
        )
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == numpy.float64

    with mp.workdps(200):
        assert G_exact(-0.001, 0, 0.2) == pytest.approx(
            G_approx(-0.001, 0, 0.2), rel=0.01
        )
        assert G_exact(-0.001, 0, 0.2) == pytest.approx(
            G_approx_with_units(Q_(-10, "um"), Q_(0, "cm"), Q_(0.2, "s")).magnitude,
            rel=0.01,
        )

    with mp.workdps(2000):
        assert G_exact(0, 0, 2) == pytest.approx(G_approx(0, 0, 2), rel=0.01)
        assert G_exact(0, 0, 2) == pytest.approx(
            G_approx_with_units(Q_(0, "cm"), Q_(0, "cm"), Q_(2, "s")).magnitude,
            rel=0.01,
        )
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == numpy.float64


def test_flat_top_beam_call_function():
    G = greens_functions.FlatTopBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "1 cm",
            "z0": "0 cm",
            "R": "1 cm",
            "with_units": True,
            "use_approximate": False,
        }
    )

    assert G(Q_(0, "cm"), Q_(0, "cm"), Q_(0, "s")).to("K/s") == Q_(1 / 2, "K/s")
    assert G(Q_(1, "cm"), Q_(0, "cm"), Q_(0, "s")).to("K/s").magnitude == pytest.approx(
        Q_(1 / 2, "K/s") * math.exp(-1) * (1 - 0)
    )
    assert G(Q_(1, "cm"), Q_(0, "cm"), Q_(1, "s")).to("K/s").magnitude == pytest.approx(
        Q_(1 / 2, "K/s").magnitude
        * math.exp(-1)
        * math.exp(1)
        * (scipy.special.erf(1) - scipy.special.erf(-1 / math.sqrt(4) + 1))
        * (1 - math.exp(-1 / 4))
    )


def test_multi_layer_greens_function_errors():
    with pytest.raises(RuntimeError) as e:
        G = greens_functions.MultiLayerGreensFunction(
            {
                "laser": {
                    "E0": "1 W/cm^2",
                    "R": "1 cm",
                },
                "thermal": {
                    "k": "1 W/cm/K",
                    "rho": "1 g/cm^3",
                    "c": "1 J/g/K",
                },
                "layers": [
                    {
                        "mua": "1 1/cm",
                        "d": "1 cm",
                        "z0": "0.5 cm",
                    },
                    {
                        "mua": "1 1/cm",
                        "d": "1 cm",
                        "z0": "0 cm",
                    },
                ],
                "with_units": False,
            }
        )


def test_multi_layer_greens_function_calcs():
    G1 = greens_functions.MultiLayerGreensFunction(
        {
            "laser": {
                "E0": "1 W/cm^2",
            },
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "1 cm",
                    "z0": "0 cm",
                },
            ],
            "with_units": False,
        }
    )
    G2 = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "1 cm",
            "z0": "0 cm",
            "with_units": False,
        }
    )

    assert G1(1, 0, 1) == pytest.approx(G2(1, 0, 1))

    G1 = greens_functions.MultiLayerGreensFunction(
        {
            "laser": {
                "E0": "1 W/cm^2",
                "R": "1 cm",
            },
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "1 cm",
                    "z0": "0 cm",
                },
            ],
            "with_units": False,
        }
    )
    G2 = greens_functions.FlatTopBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "R": "1 cm",
            "d": "1 cm",
            "z0": "0 cm",
            "with_units": False,
        }
    )

    assert G1(1, 0, 1) == pytest.approx(G2(1, 0, 1))
