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
  one_over_e_radius: $(${D}/2)
  wavelength: 530 nm

temperature_rise:
  sensor:
      z: 70 um
      r: 0 um
  use_approximations: True
  temperature_rise:
    method: quad
  output_file: 'output/CW/output-Tvst.txt'
  output_config_file: 'output/CW/output-CONFIG.yml'
  time:
      resolution: 0.1 ms
      max: 2 ms

"""
    )


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # assert "Usage: retina-therm" in result.stdout
    assert "Usage:" in result.stdout


def test_cli_simple_model_hdf5_output(simple_config):
    runner = CliRunner()
    with runner.isolated_filesystem():
        simple_config["temperature_rise"]["output_file_format"] = "hdf5"
        pathlib.Path("input.yml").write_text(yaml.dump(simple_config))
        result = runner.invoke(app, ["temperature-rise", "input.yml"])
        if result.exit_code != 0:
            print(result.stdout)
        assert result.exit_code == 0
        assert pathlib.Path("output/CW/output-Tvst.txt").exists()
        assert pathlib.Path("output/CW/output-CONFIG.yml").exists()

        with pytest.raises(UnicodeDecodeError):
            output = pathlib.Path("output/CW/output-Tvst.txt").read_text()


def test_cli_simple_model(simple_config):
    runner = CliRunner()
    with runner.isolated_filesystem():
        pathlib.Path("input.yml").write_text(yaml.dump(simple_config))
        result = runner.invoke(app, ["temperature-rise", "input.yml"])
        if result.exit_code != 0:
            print(result.stdout)
        assert result.exit_code == 0
        assert pathlib.Path("output/CW/output-Tvst.txt").exists()
        assert pathlib.Path("output/CW/output-CONFIG.yml").exists()
