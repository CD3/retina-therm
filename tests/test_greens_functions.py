import copy
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
            "use_approximations": False,
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
            "use_approximations": False,
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
            "use_approximations": False,
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
            "use_approximations": True,
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
            "use_approximations": True,
        }
    )

    assert G_approx(0, 0, 0.0) == pytest.approx(float(G_exact(0, 0, 0.0)))
    assert G_approx_with_units(
        Q_(0, "cm"), Q_(0, "cm"), Q_(0.0, "s")
    ).magnitude == pytest.approx(float(G_exact(0, 0, 0.0)))
    assert type(G_exact(0, 0, 0.0)) == mp.mpf
    assert type(G_approx(0, 0, 0.0)) == float  # numpy.float64

    assert G_approx(0, 0, 0.01) == pytest.approx(float(G_exact(0, 0, 0.01)), rel=0.03)
    assert G_approx_with_units(
        Q_(0, "cm"), Q_(0, "cm"), Q_(0.01, "s")
    ).magnitude == pytest.approx(float(G_exact(0, 0, 0.01)), rel=0.03)
    assert type(G_exact(0, 0, 0.01)) == mp.mpf
    assert type(G_approx(0, 0, 0.01)) == float  # numpy.float64

    with mp.workdps(200):
        assert G_approx(0, 0, 0.2) == pytest.approx(float(G_exact(0, 0, 0.2)), rel=0.01)
        assert G_approx_with_units(
            Q_(0, "cm"), Q_(0, "cm"), Q_(0.2, "s")
        ).magnitude == pytest.approx(
            float(G_exact(0, 0, 0.2)),
            rel=0.01,
        )
        assert type(G_exact(0, 0, 0.2)) == mp.mpf
        assert type(G_approx(0, 0, 0.2)) == float  # numpy.float64

    with mp.workdps(2000):
        assert G_approx(0, 0, 2) == pytest.approx(float(G_exact(0, 0, 2)), rel=0.01)
        assert G_approx_with_units(
            Q_(0, "cm"), Q_(0, "cm"), Q_(2, "s")
        ).magnitude == pytest.approx(
            float(G_exact(0, 0, 2)),
            rel=0.01,
        )
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == float  # numpy.float64

    with mp.workdps(200):
        assert G_approx(-0.001, 0, 0.2) == pytest.approx(
            float(G_exact(-0.001, 0, 0.2)), rel=0.01
        )
        assert G_approx_with_units(
            Q_(-10, "um"), Q_(0, "cm"), Q_(0.2, "s")
        ).magnitude == pytest.approx(
            float(G_exact(-0.001, 0, 0.2)),
            rel=0.01,
        )

    with mp.workdps(2000):
        assert G_approx(0, 0, 2) == pytest.approx(float(G_exact(0, 0, 2)), rel=0.01)
        assert G_approx_with_units(
            Q_(0, "cm"), Q_(0, "cm"), Q_(2, "s")
        ).magnitude == pytest.approx(
            float(G_exact(0, 0, 2)),
            rel=0.01,
        )
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == float  # numpy.float64


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
            "one_over_e_radius": "1 cm",
            "with_units": True,
            "use_approximations": False,
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
                        "d": "1 cm",
                        "z0": "0.5 cm",
                    },
                    {
                        "mua": "1 1/cm",
                        "d": "1 cm",
                        "z0": "0 cm",
                    },
                ],
                "simulation": {
                    "with_units": False,
                },
            }
        )


def test_multi_layer_greens_function_calcs():
    # single layer should give same thing as a the absorbing layer class
    G1 = greens_functions.MultiLayerGreensFunction(
        {
            "simulation": {
                "with_units": False,
            },
            "laser": {"E0": "1 W/cm^2", "profile": "1d"},
            "thermal": {
                "k": "1 W/cm/K",
                "rho": "1 g/cm^3",
                "c": "1 J/g/K",
            },
            "layers": [
                {
                    "mua": "1 1/cm",
                    "d": "1 cm",
                    "z0": "0.001 cm",
                },
            ],
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
            "z0": "0.001 cm",
            "with_units": False,
        }
    )

    assert G1(1, 0, 1) == pytest.approx(G2(1, 0, 1))

    assert G1(1, 0, 1) == pytest.approx(G2(1, 0, 1))

    # two layers with the same absorption coefficient should give the same answer
    # as one big layer

    one_layer_config = {
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
                "mua": "1000 1/cm",
                "d": "10 um",
                "z0": "0 um",
            },
        ],
        "simulation": {
            "with_units": False,
            "use_approximations": False,
        },
    }

    two_layer_config = copy.deepcopy(one_layer_config)
    two_layer_config.update(
        {
            "simulation": {},
            "layers": [
                {
                    "mua": "1000 1/cm",
                    "d": "5 um",
                    "z0": "0 um",
                },
                {
                    "mua": "1000 1/cm",
                    "d": "5 um",
                    "z0": "5 um",
                },
            ],
        }
    )
    G1 = greens_functions.MultiLayerGreensFunction(one_layer_config)
    G2 = greens_functions.MultiLayerGreensFunction(two_layer_config)
    with open("Tvsz.txt", "w") as f:
        for z in numpy.arange(-100 * 1e-4, 100 * 1e-4, 1e-6):
            f.write(f"{z} {G1(z,0,1e-6)} {G2(z,0,1e-6)}\n")
    # print(G1(0.00035, 0, 1e-6))
    # print(G1(0.00037, 0, 1e-6))
    assert G1(0, 0, 1e-6) == pytest.approx(G2(0, 0, 1e-6), rel=0.01)
    assert G2(0.0002, 0, 2e-7) < G2(0.00025, 0, 2e-7)
    assert G1(0.00025, 0, 2e-7) == pytest.approx(G2(0.00025, 0, 2e-7))


def test_discontinuity_bug():
    one_layer_config = {
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
                "mua": "1000 1/cm",
                "d": "10 um",
                "z0": "0 um",
            },
        ],
        "simulation": {
            "with_units": False,
            "use_approximations": False,
            "use_multi_precision": True,
        },
    }
    one_layer_config["simulation"]["use_multi_precision"] = False
    G1 = greens_functions.MultiLayerGreensFunction(one_layer_config)
    assert G1(0.00035, 0, 1e-6) < G1(0.00037, 0, 1e-6)

    one_layer_config["simulation"]["use_multi_precision"] = False
    one_layer_config["simulation"]["use_approximations"] = True
    G2 = greens_functions.MultiLayerGreensFunction(one_layer_config)
    mp.esp = 1000
    with open("Tvsz.txt", "w") as f:
        for z in numpy.arange(-100 * 1e-4, 100 * 1e-4, 1e-6):
            f.write(f"{z} {G1(z,0,10e-6)} {G2(z,0,10e-6)}\n")
    with open("Tvst.txt", "w") as f:
        for t in numpy.logspace(-10, 0, 100):
            f.write(f"{t} {G1(0,0,t)} {G1(5e-4,0,t)} {G2(0,0,t)} {G2(5e-4,0,t)}\n")
    # assert G1(0.00035, 0, 1e-6) < G1(0.00037, 0, 1e-6)


def test_derivation_bug():
    # had an error in the derivation that affected layers positioned off of z = 0.
    # i.e., if z_0 != 0, the formula was not correct and we got different answers
    # for the temperature at z = z_0
    G1 = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "100 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "10 um",
            "z0": "0 um",
            "with_units": False,
            "use_approximations": False,
            "use_multi_precision": True,
        }
    )
    G2 = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "100 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "10 um",
            "z0": "10 um",
            "with_units": False,
            "use_approximations": False,
            "use_multi_precision": True,
        }
    )

    assert G1(0, 0, 1e-8) == G2(10e-4, 0, 1e-8)
    assert G1(0, 0, 1e-3) == G2(10e-4, 0, 1e-3)
