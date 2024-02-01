import pytest
import yaml

from retina_therm.schemas import *


def test_layer_schema():
    config = {"d": "1 um", "z0": "10 um", "mua": "310 1/cm"}

    layer = Layer(**config)

    assert layer.d.magnitude == pytest.approx(1e-4)
    assert layer.z0.magnitude == pytest.approx(0.001)
    assert layer.mua.magnitude == pytest.approx(310)

    print(layer.model_dump())


def test_LargeBeamAbsorbingLayerGreensFunctionConfig():
    config_text = """
mua: 10 1/mm
rho: 1 kg/m^3
c: 1 cal / g / K
k: 2 W/cm/K
d: 10 um
z0: 0 um
E0: 1 mW/cm^2
"""

    config = LargeBeamAbsorbingLayerGreensFunctionConfig(**yaml.safe_load(config_text))
    print(config.model_dump())
