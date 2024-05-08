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
    assert type(G_approx(0, 0, 0.0)) == numpy.float64

    assert G_exact(0, 0, 0.01) == pytest.approx(G_approx(0, 0, 0.01), rel=0.001)
    assert type(G_exact(0, 0, 0.01)) == mp.mpf
    assert type(G_approx(0, 0, 0.01)) == numpy.float64

    with mp.workdps(200):
        assert G_exact(0, 0, 0.2) == pytest.approx(G_approx(0, 0, 0.2), rel=0.001)
        assert type(G_exact(0, 0, 0.2)) == mp.mpf
        assert type(G_approx(0, 0, 0.2)) == numpy.float64

    with mp.workdps(2000):
        assert G_exact(0, 0, 2) == pytest.approx(G_approx(0, 0, 2), rel=0.001)
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == numpy.float64

    with mp.workdps(200):
        assert G_exact(-0.001, 0, 0.2) == pytest.approx(
            G_approx(-0.001, 0, 0.2), rel=0.01
        )

    with mp.workdps(2000):
        assert G_exact(0, 0, 2) == pytest.approx(G_approx(0, 0, 2), rel=0.001)
        assert type(G_exact(0, 0, 2)) == mp.mpf
        assert type(G_approx(0, 0, 2)) == numpy.float64


