import contextlib
import os
import pathlib

import pytest
import yaml
from typer.testing import CliRunner

from retina_therm.cli import app

from .unit_test_utils import working_directory


@pytest.fixture
def simple_config():
    return yaml.safe_load(
        """
thermal:
    k: 0.6306 W/m/K
    rho: 992 kg/m^3
    c: 4178 J /kg / K
layers:
  - name: retina
    z0: 0 um
    d: 10 um
    mua: 100 1/cm
laser:
  E0: 1 W/cm^2
  alpha: 1.5 mrad
  D: 100 um
  wavelength: 530 nm

sensor:
  z: 70 um
  r: 0 um

simulation:
  use_approximations: True
  temperature_rise:
    method: quad
  output_file: 'output/CW/output-Tvst.txt'
  output_config_file: 'output/CW/output-CONFIG.yml'
  time:
      dt: 0.1 ms
      max: 2 ms

"""
    )


@pytest.fixture
def base_schulmeister_config():
    return yaml.safe_load(
        """
thermal:
    k: 0.6306 W/m/K
    rho: 992 kg/m^3
    c: 4178 J /kg / K
layers:
  - name: Henle's fiber layer
    z0: 0 um
    d: 7 um
    mua: models.schulmeister.mua.HenlesFiberLayer({wavelength})
  - name: Pigmented RPE
    z0: 67 um
    d: 10 um
    mua: models.schulmeister.mua.RPE({wavelength})
  - name: Choiroid
    z0: 81 um
    d: 170 um
    mua: models.schulmeister.mua.Choroid({wavelength})
laser:
  E0: 1 W/cm^2
  alpha: 1.5 mrad
  L: 17 mm
  D: $( ${L} * ${alpha}.to("rad") )
  wavelength: 530 nm


sensor:
  z: 70 um
  r: 0 um

simulation:
  use_approximations: True
  temperature_rise:
    method: quad
  output_file: 'output/CW/output-Tvst.txt'
  output_config_file: 'output/CW/output-CONFIG.yml'
  time:
      dt: 0.1 ms
      max: 2 ms

"""
    )


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # assert "Usage: retina-therm" in result.stdout
    assert "Usage:" in result.stdout


def test_cli_simple_model(simple_config):
    runner = CliRunner()
    with runner.isolated_filesystem():
        pathlib.Path("input.yml").write_text(yaml.dump(simple_config))
        result = runner.invoke(app, ["temperature-rise", "input.yml"])
        assert result.exit_code == 0
        assert pathlib.Path("output/CW/output-Tvst.txt").exists()
        assert pathlib.Path("output/CW/output-CONFIG.yml").exists()

        simple_config["simulation"]["output_file"] = "{c[laser/D]}-Tvst.txt"
        pathlib.Path("input.yml").write_text(yaml.dump(simple_config))
        result = runner.invoke(app, ["temperature-rise", "input.yml"])
        assert result.exit_code == 0
        assert pathlib.Path("100_micrometer-Tvst.txt").exists()

        output = pathlib.Path("output/CW/output-Tvst.txt").read_text()


def test_cli_schulmeister_model(base_schulmeister_config):
    runner = CliRunner()
    with runner.isolated_filesystem():
        pathlib.Path("input.yml").write_text(yaml.dump(base_schulmeister_config))
        result = runner.invoke(app, ["temperature-rise", "input.yml"])
        assert result.exit_code == 0
        assert pathlib.Path("output/CW/output-Tvst.txt").exists()
        assert pathlib.Path("output/CW/output-CONFIG.yml").exists()


def test_cli_simple_model(simple_config):
    runner = CliRunner()
    with runner.isolated_filesystem():
        simple_config["simulation"]["output_format"] = "hdf5"
        pathlib.Path("input.yml").write_text(yaml.dump(simple_config))
        result = runner.invoke(app, ["temperature-rise", "input.yml"])
        assert result.exit_code == 0
        assert pathlib.Path("output/CW/output-Tvst.txt").exists()
        assert pathlib.Path("output/CW/output-CONFIG.yml").exists()

        with pytest.raises(UnicodeDecodeError):
            output = pathlib.Path("output/CW/output-Tvst.txt").read_text()
