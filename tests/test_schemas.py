import pydantic
import pytest
import yaml

from retina_therm.schemas import *
from retina_therm.units import *


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
    laser_config = Laser(**{"profile": "gaussian", "R": "1 cm", "E0": "1 W/cm^2"})
    assert laser_config.profile == "gaussian"

    laser_config = Laser(**{"profile": "flattop", "R": "1 cm", "E0": "1 W/cm^2"})
    assert laser_config.profile == "flattop"

    laser_config = Laser(**{"profile": "Gaussian", "R": "1 cm", "E0": "1 W/cm^2"})
    assert laser_config.profile == "gaussian"

    laser_config = Laser(**{"profile": "Flat Top", "R": "1 cm", "E0": "1 W/cm^2"})
    assert laser_config.profile == "flattop"

    with pytest.raises(Exception):
        laser_config = Laser(**{"profile": "annular", "R": "1 cm", "E0": "1 W/cm^2"})

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
            "simulation": {},
            "laser": {"profile": "Flat Top", "R": "10 um", "E0": "1 W/cm^2"},
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


def test_LaserSchema():
    laser_config = Laser(**{"profile": "gaussian", "R": "1 cm", "E0": "1 W/cm^2"})
    assert laser_config.D.magnitude == pytest.approx(2)
    laser_config.D = Q_(1, "cm")
    assert laser_config.R.magnitude == pytest.approx(0.5)

    laser_config = Laser(**{"profile": "gaussian", "D": "1 cm", "E0": "1 W/cm^2"})
    assert laser_config.R.magnitude == pytest.approx(0.5)

    with pytest.raises(Exception):
        laser_config = Laser(**{"profile": "gaussian", "q": "1 cm"})


def test_converting_models():
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
    config2 = FlatTopBeamAbsorbingLayerGreensFunctionConfig(**config.model_dump())
    config3 = LargeBeamAbsorbingLayerGreensFunctionConfig(**config2.model_dump())

    assert config.R == config2.R
    with pytest.raises(AttributeError):
        tmp = config3.R


def test_unit_conversions():
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

    assert config.mua.magnitude == pytest.approx(100)
    assert config.mua.units == Q_("1 1/cm").units

    assert config.rho.magnitude == pytest.approx(
        Q_(1, "kg/m^3").to("g/cm/cm/cm").magnitude
    )
    assert config.rho.units == Q_("1 g/cm^3").units

    assert config.c.magnitude == pytest.approx(Q_(1, "cal/g/K").to("J/g/K").magnitude)
    assert config.c.units == Q_("1 J/g/K").units

    assert config.k.magnitude == pytest.approx(2)
    assert config.k.units == Q_("1 W/cm/K").units

    assert config.d.magnitude == pytest.approx(10e-4)
    assert config.d.units == Q_("1 cm").units

    assert config.z0.magnitude == pytest.approx(0)
    assert config.z0.units == Q_("1 cm").units

    assert config.E0.magnitude == pytest.approx(0.001)
    assert config.E0.units == Q_("1 W/cm/cm").units

    assert config.R.magnitude == pytest.approx(0.3)
    assert config.R.units == Q_("1 cm").units


def test_beam_radius_required_if_not_1d():
    with pytest.raises(pydantic.ValidationError):
        config = Laser(**{"E0": 1})
    with pytest.raises(pint.DimensionalityError):
        config = Laser(**{"E0": "1"})

    with pytest.raises(pydantic.ValidationError):
        config = Laser(**{"E0": "1 W/cm/cm"})

    config = Laser(**{"E0": "1 W/cm/cm", "R": "1 cm"})

    assert config.profile == "flattop"
    assert config.R.magnitude == pytest.approx(1)

    config = Laser(**{"E0": "1 W/cm/cm", "profile": "1d"})

    assert config.profile == "1d"
    assert config.R == None


def test_cw_retina_exposure_config():
    config = CWRetinaLaserExposureConfig(
        **{
            "simulation": {},
            "laser": {"profile": "Flat Top", "R": "10 um", "E0": "1 W/cm^2"},
            "thermal": {"rho": "1 g/cm^3", "c": "1 cal/g/K", "k": "0.00628 W/cm/K"},
            "layers": [
                {"d": "10 um", "z0": "0 um", "mua": "300 1/cm"},
                {"d": "100 um", "z0": "10 um", "mua": "50 1/cm"},
            ],
        }
    )

    assert config.laser.start.magnitude == pytest.approx(0)
    assert config.laser.duration.magnitude == pytest.approx(Q_(1, "year").to("s"))


def test_pulsed_retina_exposure_config():
    config_dict = {
        "simulation": {},
        "laser": {"profile": "Flat Top", "R": "10 um", "E0": "1 W/cm^2"},
        "thermal": {"rho": "1 g/cm^3", "c": "1 cal/g/K", "k": "0.00628 W/cm/K"},
        "layers": [
            {"d": "10 um", "z0": "0 um", "mua": "300 1/cm"},
            {"d": "100 um", "z0": "10 um", "mua": "50 1/cm"},
        ],
    }
    with pytest.raises(pydantic.ValidationError):
        config = PulsedRetinaLaserExposureConfig(**config_dict)

    config_dict["laser"]["pulse_duration"] = "1 us"
    config = PulsedRetinaLaserExposureConfig(**config_dict)
    assert config.laser.start.magnitude == pytest.approx(0)
    assert config.laser.duration.magnitude == pytest.approx(
        Q_(1, "year").to("s").magnitude
    )
    assert config.laser.pulse_duration.magnitude == pytest.approx(1e-6)
    assert config.laser.pulse_period.magnitude == pytest.approx(
        Q_(1, "year").to("s").magnitude
    )

    config_dict["laser"]["pulse_period"] = "10 us"
    config = PulsedRetinaLaserExposureConfig(**config_dict)
    assert config.laser.start.magnitude == pytest.approx(0)
    assert config.laser.duration.magnitude == pytest.approx(
        Q_(1, "year").to("s").magnitude
    )
    assert config.laser.pulse_duration.magnitude == pytest.approx(1e-6)
    assert config.laser.pulse_period.magnitude == pytest.approx(10e-6)


def test_multiple_pulse_cmd_input():

    MultiplePulseContribution(**{"arrival_time": "0 s", "scale": 1.2})

    input = {
        "input_file": "",
        "output_file": "",
        "output_config_file": "",
        "t0": "1 s",
        "N": 20,
        "scale": 2,
    }

    MultiplePulseCmdConfig(**input)

    input = {
        "input_file": "",
        "output_file": "",
        "output_config_file": "",
        "contributions": [{"arrival_time": "0 s", "scale": 1.1}],
    }

    MultiplePulseCmdConfig(**input)

    with pytest.raises(ValueError) as e:
        input = {
            "input_file": "",
            "output_file": "",
            "output_config_file": "",
        }

        MultiplePulseCmdConfig(**input)
    assert "'t0'" in str(e)
    assert "'N'" in str(e)
