import pytest
import yaml

from retina_therm.schemas import *


def test_layer_schema():
    config = {"d": "1 um", "z0": "10 um", "mua": "310 1/cm"}

    layer = Layer(**config)

    assert layer.d.magnitude == pytest.approx(1e-4)
    assert layer.z0.magnitude == pytest.approx(0.001)
    assert layer.mua.magnitude == pytest.approx(310)

    # print(layer.model_dump())


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
    assert config.E0.magnitude == 0.001
    # print(config.model_dump())


def test_FlatTopBeamAbsorbingLayerGreensFunctionConfig():
    config_text = """
mua: 10 1/mm
rho: 1 kg/m^3
c: 1 cal / g / K
k: 2 W/cm/K
d: 10 um
z0: 0 um
E0: 1 mW/cm^2
R: 3 mm
"""

    config = FlatTopBeamAbsorbingLayerGreensFunctionConfig(
        **yaml.safe_load(config_text)
    )
    # print(config.model_dump())
    assert config.R.magnitude == pytest.approx(0.3)


def test_GaussianBeamAbsorbingLayerGreensFunctionConfig():
    config_text = """
mua: 10 1/mm
rho: 1 kg/m^3
c: 1 cal / g / K
k: 2 W/cm/K
d: 10 um
z0: 0 um
E0: 1 mW/cm^2
R: 3 mm
"""

    config = GaussianBeamAbsorbingLayerGreensFunctionConfig(
        **yaml.safe_load(config_text)
    )
    # print(config.model_dump())
    assert config.R.magnitude == pytest.approx(0.3)


def test_MultiLayerGreensFunctionConfig():
    laser_config = Laser(**{"profile": "gaussian", "R": "1 cm"})
    assert laser_config.profile == "gaussian"

    laser_config = Laser(**{"profile": "flattop", "R": "1 cm"})
    assert laser_config.profile == "flattop"

    laser_config = Laser(**{"profile": "Gaussian", "R": "1 cm"})
    assert laser_config.profile == "gaussian"

    laser_config = Laser(**{"profile": "Flat Top", "R": "1 cm"})
    assert laser_config.profile == "flattop"

    with pytest.raises(Exception):
        laser_config = Laser(**{"profile": "annular", "R": "1 cm"})

    thermal_config = ThermalProperties(
        **{"rho": "1 g/cm**3", "c": "1 cal/g/K", "k": "1 W/cm/K"}
    )
    assert thermal_config.rho.magnitude == pytest.approx(1)
    assert thermal_config.c.magnitude == pytest.approx(4.184)
    assert thermal_config.k.magnitude == pytest.approx(1)

    layer_config = Layer(**{"d": "100 um", "z0": "10 um", "mua": "300 1/cm"})
    assert layer_config.d.magnitude == pytest.approx(100e-4)
    assert layer_config.z0.magnitude == pytest.approx(10e-4)
    assert layer_config.mua.magnitude == pytest.approx(300)

    config = MultiLayerGreensFunctionConfig(
        **{
            "laser": {"profile": "Flat Top", "R": "10 um"},
            "thermal": {"rho": "1 g/cm^3", "c": "1 cal/g/K", "k": "0.00628 W/cm/K"},
            "layers": [
                {"d": "10 um", "z0": "0 um", "mua": "300 1/cm"},
                {"d": "100 um", "z0": "10 um", "mua": "50 1/cm"},
            ],
        }
    )

    assert config.laser.profile == "flattop"
    assert config.thermal.rho.magnitude == pytest.approx(1)
    assert config.layers[0].d.magnitude == pytest.approx(10e-4)
