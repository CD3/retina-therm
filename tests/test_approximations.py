import math

from mpmath import mp

mp.dsp = 1000

import numpy
import pytest
import scipy

from retina_therm import greens_functions
from retina_therm.units import *


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
    assert G_exact(0, 0, 0.0) == pytest.approx(G_approx(0, 0, 0.0))
    assert type(G_exact(0, 0, 0.0)) == mp.mpf
    assert type(G_approx(0, 0, 0.0)) == float  # numpy.float64

    assert G_exact(0, 0, 0.01) == pytest.approx(G_approx(0, 0, 0.01), rel=0.001)
    assert type(G_exact(0, 0, 0.01)) == mp.mpf
    assert type(G_approx(0, 0, 0.01)) == float  # numpy.float64

    with mp.workdps(200):
        assert G_exact(0, 0, 0.2) == pytest.approx(G_approx(0, 0, 0.2), rel=0.001)
        assert type(G_exact(0, 0, 0.2)) == mp.mpf
        assert type(G_approx(0, 0, 0.2)) == float  # numpy.float64

    with mp.workdps(2000):
        assert G_exact(0, 0, 2) == pytest.approx(G_approx(0, 0, 2), rel=0.001)
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == float  # numpy.float64

    with mp.workdps(200):
        assert G_exact(-0.001, 0, 0.2) == pytest.approx(
            G_approx(-0.001, 0, 0.2), rel=0.01
        )

    with mp.workdps(2000):
        assert G_exact(0, 0, 2) == pytest.approx(G_approx(0, 0, 2), rel=0.001)
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == float  # numpy.float64

    with open("Tvsz-exact_vs_approx.txt", "w") as f:
        for z in numpy.arange(-100 * 1e-4, 100 * 1e-4, 1e-6):
            f.write(f"{z} {G_exact(z,0,1000e-6)} {G_approx(z,0,1000e-6)}\n")


def test_axial_part_fast_heat_flow_long_time_approx():
    G_exact_mp = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1200 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "12 um",
            "z0": "0 cm",
            "with_units": False,
            "use_multi_precision": True,
            "use_approximations": False,
        }
    )
    G_exact = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1200 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "12 um",
            "z0": "0 cm",
            "with_units": False,
            "use_multi_precision": False,
            "use_approximations": False,
        }
    )
    G_approx = greens_functions.LargeBeamAbsorbingLayerGreensFunction(
        {
            "mua": "1200 1/cm",
            "k": "1 W/cm/K",
            "rho": "1 g/cm^3",
            "c": "1 J/g/K",
            "E0": "1 W/cm^2",
            "d": "12 um",
            "z0": "0 cm",
            "with_units": False,
            "use_multi_precision": False,
            "use_approximations": True,
        }
    )

    # for this configuration, there are certain positions were
    # the conditions for "long time" is met, but the approximation is inaccurate.
    # we have identified a few of these points in order to write a test case that will fail

    assert G_approx(0.0006, 0, 1e-6) == pytest.approx(G_exact(0.0006, 0, 1e-6))
    assert G_approx(0.00065, 0, 1e-6) == pytest.approx(G_exact(0.00065, 0, 1e-6))
    assert G_approx(0.0006, 0, 1e-5) == pytest.approx(G_exact(0.0006, 0, 1e-5))
    assert G_approx(0.00065, 0, 1e-5) == pytest.approx(G_exact(0.00065, 0, 1e-5))
    assert G_approx(0.0006, 0, 2e-5) == pytest.approx(
        G_exact(0.0006, 0, 2e-5), rel=0.02
    )
    assert G_approx(0.00065, 0, 2e-5) == pytest.approx(
        G_exact(0.00065, 0, 2e-5), rel=0.02
    )

    assert G_exact(0.0006, 0, 3e-5) == 0
    assert G_exact(0.00065, 0, 3e-5) == 0

    assert G_approx(0.0006, 0, 3e-5) > 0
    assert G_approx(0.00065, 0, 3e-5) > 0

    # note that we need more and more precisions to calculate temperatures at longer and longer
    # times without an approximation.
    with mp.workdps(200):
        assert G_approx(0.0006, 0, 3e-5) == pytest.approx(
            float(G_exact_mp(0.0006, 0, 3e-5)), rel=0.02
        )
        assert G_approx(0.00065, 0, 3e-5) == pytest.approx(
            float(G_exact_mp(0.00065, 0, 3e-5)), rel=0.02
        )

    with mp.workdps(1000):
        assert G_approx(0.0006, 0, 1e-3) == pytest.approx(
            float(G_exact_mp(0.0006, 0, 1e-3)), rel=0.01
        )
        assert G_approx(0.00065, 0, 1e-3) == pytest.approx(
            float(G_exact_mp(0.00065, 0, 1e-3)), rel=0.01
        )

    # with open("Tvsz-exact_vs_approx-fast.txt", "w") as f:
    #     for z in numpy.arange(-100 * 1e-4, 100 * 1e-4, 1e-6):
    #         f.write(f"{z} {G_exact(z,0,1e-6)} {G_approx(z,0,1e-6)}\n")
