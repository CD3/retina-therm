import contextlib
import os
import pathlib

import pytest
import yaml
from typer.testing import CliRunner

from retina_therm import cli
from retina_therm.cli import app, compute_job_split

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
  method: quad
  output_file: 'output/CW/output-Tvst.txt'
  output_config_file: 'output/CW/output-CONFIG.yml'
  time:
      resolution: 0.1 ms
      max: 20 ms

"""
    )


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # assert "Usage: retina-therm" in result.stdout
    assert "Usage:" in result.stdout


@pytest.mark.skip(
    reason="Getting error about `isolated_filesystem` not being an attribute of CliRunner...."
)
@pytest.mark.timeout(5)
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


@pytest.mark.skip(
    reason="Getting error about `isolated_filesystem` not being an attribute of CliRunner...."
)
@pytest.mark.timeout(5)
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


@pytest.mark.skip(
    reason="Getting error about `isolated_filesystem` not being an attribute of CliRunner...."
)
@pytest.mark.timeout(5)
def test_cli_config_error_in_subprocess(simple_config):
    runner = CliRunner()
    with runner.isolated_filesystem():
        simple_config["thermal"]["k"] = "$(${/missing})"
        pathlib.Path("input.yml").write_text(yaml.dump(simple_config))
        result = runner.invoke(app, ["temperature-rise", "input.yml"])
        if result.exit_code != 0:
            print(result.stdout)
        assert result.exit_code != 0


@pytest.mark.parametrize(
    "njobs,num_configs,cpu_count,expected",
    [
        # large batch: main jobs capped at cpu count, not one per config
        (None, 1002, 16, (16, 1)),
        # --njobs N is honored for large batches
        ("4", 1002, 16, (4, 1)),
        # few configs: spare cpus are used for sub jobs
        (None, 2, 16, (2, 8)),
        ("8", 3, 16, (3, 2)),
        (None, 1, 16, (1, 16)),
        # explicit A:B is passed through
        ("2:3", 1002, 16, (2, 3)),
    ],
)
def test_compute_job_split(njobs, num_configs, cpu_count, expected):
    assert compute_job_split(njobs, num_configs, cpu_count) == expected


def test_compute_job_split_bounds():
    for cpu_count in [1, 2, 7, 16, 64]:
        for num_configs in range(1, 2000):
            main, sub = compute_job_split(None, num_configs, cpu_count)
            assert 1 <= main <= min(cpu_count, num_configs)
            assert sub >= 1


class FakeBatchJobController:
    """Records the arguments it was created with instead of starting processes."""

    instances = []

    def __init__(self, proc_type, *, njobs, args={}):
        self.njobs = njobs
        self.args = args
        self.status = cli.parallel_jobs.Signal()
        self.progress = cli.parallel_jobs.Signal()
        FakeBatchJobController.instances.append(self)

    def start(self):
        pass

    def run_jobs(self, jobs):
        return [None] * len(jobs)

    def stop(self):
        pass

    def wait(self):
        pass


def test_truncate_temperature_history_file_njobs_is_int(monkeypatch):
    FakeBatchJobController.instances = []
    monkeypatch.setattr(
        cli.parallel_jobs, "BatchJobController", FakeBatchJobController
    )
    runner = CliRunner()
    result = runner.invoke(
        app, ["truncate-temperature-history-file", "a.txt", "b.txt", "--njobs", "2"]
    )
    assert result.exit_code == 0, result.output
    assert len(FakeBatchJobController.instances) == 1
    assert FakeBatchJobController.instances[0].njobs == 2
