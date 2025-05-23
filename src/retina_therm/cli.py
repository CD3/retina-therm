import copy
import importlib
import itertools
import multiprocessing
import pprint
import shutil
import subprocess
import sys
from pathlib import Path, PosixPath
from typing import Annotated, List, Literal, Optional, Union

import numpy
import powerconf
import rich
import scipy
import typer
import yaml
from fspathtree import fspathtree
from mpmath import mp
from pydantic import BeforeValidator, ValidationError
from tqdm import tqdm

import retina_therm
from retina_therm import (config, greens_functions, multi_pulse_builder,
                          signals, units, utils)

from . import parallel_jobs, utils

__version__ = importlib.metadata.version("retina-therm")


app = typer.Typer()
console = rich.console.Console()


def path_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))


yaml.add_representer(PosixPath, path_representer)


def version_callback(value: bool):
    if value:
        typer.echo(f"retina-therm: {__version__}")
        raise typer.Exit()


def q2str(p, v):
    if hasattr(v, "magnitude"):
        return str(v)
    return v


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
):
    pass


def compute_evaluation_times(config):
    # if times are given in the config, just them
    if "ts" in config:
        t = numpy.array([units.Q_(time).to("s").magnitude for time in config["ts"]])
    else:
        # we want to support specifying the times as a single range,
        # i.e. "from tmin to tmax by steps of dt"
        # or multiple ranges
        # i.e. "from tmin_1 to tmax_1 by steps of dt_1 AND from tmin_2 to tmax_2 by steps of dt_2"
        # this is usefull for sampling the start of a long exposure at higher resolution than the end.
        time_configs = []
        if type(config.tree) == dict:
            time_configs.append(config)
        else:
            for c in config:
                time_configs.append(c)

        time_arrays = []
        for i, time_config in enumerate(time_configs):
            dt = units.Q_(time_config.get("resolution", "1 us"))
            # if tmin is given, use it
            # if it is not given and this is the first config, use 0 s
            # if it is not given and this is not the first config, use the last config's tmax plus our dt
            #     if the previous config does not have a tmax, use 10 s...
            tmin = units.Q_(
                time_config.get(
                    "min",
                    (
                        units.Q_(time_configs[i - 1].get("max", "10 second")) + dt
                        if i > 0
                        else "0 second"
                    ),
                )
            )
            tmax = units.Q_(time_config.get("max", "10 second"))

            dt = dt.to("s").magnitude
            tmin = tmin.to("s").magnitude
            tmax = tmax.to("s").magnitude

            # adding dt/2 here so that tmax will be included in the array
            t = numpy.arange(tmin, tmax + dt / 2, dt)
            time_arrays.append(t)
        t = numpy.concatenate(time_arrays)

    return t


temperature_rise_integration_methods = ["quad", "trap"]


def compute_tissue_properties(config):
    """
    Loops through all tissue property config keys and checks if parameter
    was given as a model instead of a specific value. If so, we call the model
    and replace the parameter value with the result of model.
    """
    for layer in config.get("layers", []):
        if "{wavelength}" in layer["mua"]:
            if "laser/wavelength" not in config:
                raise RuntimeError(
                    "Config must include `laser/wavelength` to compute absorption coefficient."
                )
            mua = eval(
                layer["mua"].format(wavelength="'" + config["/laser/wavelength"] + "'")
            )
            layer["mua"] = str(mua)  # config validators expect strings for quantities
    return config


#  _____                                   _                  ____  _
# |_   _|__ _ __ ___  _ __   ___ _ __ __ _| |_ _   _ _ __ ___|  _ \(_)___  ___
#   | |/ _ \ '_ ` _ \| '_ \ / _ \ '__/ _` | __| | | | '__/ _ \ |_) | / __|/ _ \
#   | |  __/ | | | | | |_) |  __/ | | (_| | |_| |_| | | |  __/  _ <| \__ \  __/
#   |_|\___|_| |_| |_| .__/ \___|_|  \__,_|\__|\__,_|_|  \___|_| \_\_|___/\___|
#                    |_|


class SensorConfig(config.BaseModel):
    z: config.QuantityWithUnit("cm")
    r: config.QuantityWithUnit("cm")


class TemperatureRiseConfig(config.BaseModel):
    output_file: Path
    output_config_file: Path
    output_file_format: Optional[Literal["txt"] | Literal["hdf5"]] = None
    sensor: SensorConfig
    method: Optional[Literal["trap"] | Literal["quad"]] = "quad"

    class TimeConfig(config.BaseModel):
        max: config.QuantityWithUnit("s")
        resolution: config.QuantityWithUnit("s")

    time: Optional[TimeConfig | List[TimeConfig]] = None


class TemperatureRiseCmdConfig(config.BaseModel):
    temperature_rise: TemperatureRiseConfig
    laser: config.LaserConfig
    layers: list[config.LayerConfig]
    thermal: config.ThermalPropertiesConfig


class TemperatureRiseGreensFunctionProcess(parallel_jobs.JobProcessorBase):
    """
    For running green's functino calculations in a separate process.

    This will return time-temperature pairs, it will _not_ write to files.
    """

    def __init__(self):
        super().__init__()

    def run_job(self, config):
        # Greens function classes expect simulation config params to be in /simulation
        config["/simulation"] = config["/temperature_rise"].tree
        G = greens_functions.CWRetinaLaserExposure(config.tree)
        z = config["/temperature_rise/sensor/z"]
        z = units.Q_(z).to("cm").magnitude
        r = config["/temperature_rise/sensor/r"]
        r = units.Q_(r).to("cm").magnitude

        # times are already computed by the parent process, we just need to grab them.
        t = config["/temperature_rise/time/ts"]
        self.status.emit("Computing temperature rise")
        G.progress.connect(lambda i, n: self.progress.emit(i, n))
        T = G.temperature_rise(z, r, t, method=config["/temperature_rise/method"])
        self.status.emit("done")

        return list(zip(t, T))


class TemperatureRiseSingleConfigProcess(parallel_jobs.JobProcessorBase):
    """
    For running a full simulation in a separate process.

    This uses the TemperatureRiseGreensFunctionProcess class to do the acual
    calculations, collects the time-temperature pairs and writes them to the
    output_file given in the configuration. Also writes the output_config_file.
    """

    def __init__(self, njobs=1):
        super().__init__()
        self.njobs = njobs
        self.controller = None

    def _start(self):  # Runs in CHILD
        if self.controller is None:
            self.controller = parallel_jobs.BatchJobController(
                TemperatureRiseGreensFunctionProcess, njobs=self.njobs
            )
            self.controller.start()
        self.jobs_progress = [[0, 1]] * self.njobs
        self.total_progress = [0, 1]

        def update_jobs_progress(proc, prog):
            self.jobs_progress[proc] = prog

        def compute_total_progress():
            a = 0
            b = 0
            for p in self.jobs_progress:
                a += p[0]
                b += p[1]
            return [a, b]

        # self.controller.status.connect(
        #     lambda proc, msg: parallel_jobs.pprint(">>>>", msg)
        # )
        self.controller.progress.connect(
            lambda proc, prog: update_jobs_progress(proc, prog)
        )
        self.controller.progress.connect(
            lambda *args: self.progress.emit(*compute_total_progress())
        )

    def _stop(self):  # Runs in CHILD
        if self.controller is not None:
            self.controller.stop()
            self.controller.wait()

    def run_job(self, config):  # Runs in CHILD

        # check if output files exist
        output_paths = {}
        for k in [
            "output_file",
            "output_config_file",
        ]:
            filename = config["/temperature_rise"][k]
            if filename is not None:
                path = Path(filename)
                output_paths[k + "_path"] = path
                if path.parent != Path():
                    path.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_paths[k + "_path"] = None

        if config.get("/skip_existing_outputs", False):
            if all(map(lambda k: output_paths[k].exists(), output_paths)):
                self.status.emit("Output files already exists. Skipping.")
                return

        # split the configuration up into multiple configurations over sub-intervals of the time range
        t = compute_evaluation_times(config["/temperature_rise/time"])
        t_chunks = numpy.array_split(t, self.njobs)
        configs = []
        for chunk in t_chunks:
            c = copy.deepcopy(config)
            c["/temperature_rise/time"] = {"ts": chunk}
            configs.append(c)

        # run the configurations, blocking
        results = self.controller.run_jobs(configs)

        # results will be returned in a list of lists
        # each process returns results into corresponding index of top-level list.
        # each element is a list that contains all results returned by the process,
        # one item for each time the process ran a job.
        data = list(itertools.chain(*results))
        if len(t) != len(data):  # sanity check...
            raise RuntimeError(
                f"Something went wrong. The number of computed temperature returned by subprocesses ({len(data)}) does not match the number of time points ({len(t)})"
            )
        T = numpy.array(list(map(lambda item: item[1], data)))

        self.status.emit("Writing output files...")

        if output_paths["output_config_file_path"] is not None:
            output_paths["output_config_file_path"].write_text(yaml.dump(config.tree))

        fmt = config["/temperature_rise/output_file_format"]
        if fmt is None:
            fmt = output_paths["output_file_path"].suffix[1:]
        if fmt is None:
            fmt = "txt"

        utils.write_to_file(output_paths["output_file_path"], numpy.c_[t, T], fmt)
        self.status.emit("done")


@app.command()
def temperature_rise(
    config_file: Path,
    njobs: Annotated[str, typer.Option(help="Number of parallel jobs to run.")] = None,
    dps: Annotated[
        int,
        typer.Option(help="The precision to use for calculations when mpmath is used."),
    ] = 100,
    list_methods: Annotated[
        bool, typer.Option(help="List the avaiable integration methods.")
    ] = False,
    skip_existing_outputs: Annotated[
        bool,
        typer.Option(
            help="Don't run simulation if the output file that would be written already exists."
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option(help="Print extra information")] = False,
    quiet: Annotated[bool, typer.Option(help="Don't print to console.")] = False,
):
    if list_methods:
        print("Available inegration methods:")
        for m in temperature_rise_integration_methods:
            print("  ", m)
        raise typer.Exit(0)

    mp.dps = dps

    iconsole = rich.console.Console(stderr=False, quiet=quiet)
    vconsole = rich.console.Console(
        stderr=False, quiet=True if quiet or not verbose else False
    )
    econsole = rich.console.Console(stderr=True)

    try:
        # we need to convert all quantities to strings because they will ahve been created
        # with a different unit registry than the models are using.
        configs = powerconf.yaml.powerload(
            config_file, njobs=multiprocessing.cpu_count(), transform=q2str
        )
    except KeyError as e:
        econsole.print(
            "[red]A configuration parameter references another non-existent parameter.[/red]"
        )

        econsole.print("\n\n[red]" + str(e) + "[/red]\n\n")
        raise typer.Exit(1)

    # validate configs
    for i, config in enumerate(configs):
        try:
            # validate the input config using pydantic model
            config = TemperatureRiseCmdConfig(**config.tree)
            # copy model back to config so that default values will be filled in
            configs[i] = fspathtree(config.model_dump())

        except ValidationError as e:
            econsole.print(
                "[red]There was an error reading the configuration file.[/red]"
            )
            econsole.print("\n\nPydantic Error Message:")
            econsole.print(e)
            econsole.print("\n\n")
            raise typer.Exit(1)

    # filter repeated configs
    # this allows a configuration file to contain configs for multiple commands
    # (i.e. temperature-rise and multiple-pulse) with batch parameters. If a batch
    # parameter for a different command configuration producec multiple config instances,
    # but they are all the same for this command, we only want to run one
    configs_to_run = []
    config_ids = []
    for config in configs:
        if powerconf.utils.get_id(config) not in config_ids:
            configs_to_run.append(config)
            config_ids.append(powerconf.utils.get_id(config))

    if len(configs_to_run) != len(configs):
        iconsole.print(
            "WARNING: There duplicate configurations that will be skipped. If you did not expect this, check that your batch configurations render to different instances."
        )
        configs = configs_to_run

    if skip_existing_outputs:
        for c in configs:
            c["/skip_existing_outputs"] = True

    # determine how to split up work.
    # we have a TemperatureRiseSingleConfigProcess that runs a single simulation.
    # internally, it will split the work up into chunks and use subprocesses to do the computation.
    num_jobs = multiprocessing.cpu_count()
    num_main_jobs = len(configs)
    num_sub_jobs = int(num_jobs / num_main_jobs)

    if njobs is not None:
        if ":" in njobs:
            num_main_jobs, num_sub_jobs = list(map(int, njobs.split(":")))
        else:
            num_jobs = int(njobs)
            num_main_jobs = len(configs)
            num_sub_jobs = int(num_jobs / num_main_jobs)

    controller = parallel_jobs.BatchJobController(
        TemperatureRiseSingleConfigProcess,
        njobs=num_main_jobs,
        args={"njobs": num_sub_jobs},
    )
    controller.start()

    progress_display = (
        parallel_jobs.SilentProgressDisplay()
        if quiet
        else parallel_jobs.ProgressDisplay()
    )
    progress_display.setup_new_bar("Total")
    progress_display.set_total("Total", len(configs))
    for i in range(num_main_jobs):
        progress_display.setup_new_bar(f"Job-{i:03}")
    for i in range(num_main_jobs):
        progress_display.set_progress(f"Job-{i:03}", 0, 1)

    controller.progress.connect(
        lambda proc, prog: progress_display.set_progress(f"Job-{proc:03}", *prog)
    )
    controller.status.connect(
        lambda proc, msg: (
            progress_display.update_progress("Total") if msg == "done" else None
        )
    )

    # controller.status.connect(lambda *args: print("STATUS", args))
    # controller.progress.connect(lambda *args: print("PROGRESS", args))
    results = controller.run_jobs(configs)
    controller.stop()
    controller.wait()

    raise typer.Exit(0)


#  __  __       _ _   _       _                        _
# |  \/  |_   _| | |_(_)_ __ | | ___       _ __  _   _| |___  ___
# | |\/| | | | | | __| | '_ \| |/ _ \_____| '_ \| | | | / __|/ _ \
# | |  | | |_| | | |_| | |_) | |  __/_____| |_) | |_| | \__ \  __/
# |_|  |_|\__,_|_|\__|_| .__/|_|\___|     | .__/ \__,_|_|___/\___|
#                      |_|                |_|


class PulseConfig(config.BaseModel):
    arrival_time: config.QuantityWithUnit("s")
    duration: config.QuantityWithUnit("s")
    scale: float


class MultiplePulseConfig(config.BaseModel):
    input_file: Path
    output_file: Path
    output_file_format: Optional[Literal["txt"] | Literal["hdf5"]] = None
    output_config_file: Path
    pulses: list[PulseConfig]

    class TimeConfig(config.BaseModel):
        max: Optional[config.QuantityWithUnit("s")] = None
        resolution: Optional[config.QuantityWithUnit("s")] = None

    time: Optional[TimeConfig] = TimeConfig()


class MultiplePulseCmdConfig(config.BaseModel):
    multiple_pulse: MultiplePulseConfig


class MultiplePulseProcess(parallel_jobs.JobProcessorBase):
    def run_job(self, config):

        # check if output paths exist
        output_paths = {}
        for k in ["output_file", "output_config_file"]:
            filename = config["/multiple_pulse"][k]
            output_paths[k + "_path"] = Path("/dev/stdout")
            if filename is not None:
                path = Path(filename)
                output_paths[k + "_path"] = path
                if path.parent != Path():
                    path.parent.mkdir(parents=True, exist_ok=True)

        if config.get("/skip_existing_outputs", False):
            if all(map(lambda k: output_paths[k].exists(), output_paths)):
                self.status.emit("Output files already exists. Skipping.")
                return

        self.status.emit(
            "Loading base temperature history for building multiple-pulse history."
        )
        input_file = Path(config["/multiple_pulse/input_file"])
        data = utils.read_from_file(
            input_file,
            config.get("/multiple_pulse/input_file_format", input_file.suffix[1:]),
        )

        self.status.emit("Resampling temeprature history to regularized grid")
        imax = len(data)
        tmax = units.Q_(data[-1, 0], "s")
        # if tmax is given in the config file, we want to trucate
        # the input data to include the first time >= tmax
        # this is an optimization reduces the size of the array we
        # are working.
        if config["/multiple_pulse/time/max"] is not None:
            tmax = units.Q_(config["/multiple_pulse/time/max"])
            if tmax.to("s").magnitude < data[0, 0]:
                raise RuntimeError(
                    f"/tmax ({tmax}) cannot be less than first time in history ({data[0,0]})."
                )
            if tmax.to("s").magnitude < data[-1, 0]:
                while imax > 0 and data[imax - 1, 0] > tmax.to("s").magnitude:
                    imax -= 1
        if imax < len(data):
            data = data[:imax, :]

        t = data[:, 0]
        T = data[:, 1]

        # regularize the time samples.
        # need times to be uniformly spaced apart.
        resolution = t[1] - t[0]
        if config["/multiple_pulse/time/resolution"] is not None:
            resolution = (
                units.Q_(config["/multiple_pulse/time/resolution"]).to("s").magnitude
            )

        if not multi_pulse_builder.is_resolution(t, resolution):
            tp = multi_pulse_builder.regularize_grid(t, resolution)
            Tp = multi_pulse_builder.interpolate_temperature_history(t, T, tp)
            t = tp
            T = Tp
            data = numpy.zeros([len(tp), 2])
            data[:, 0] = t

        builder = multi_pulse_builder.MultiPulseBuilder()
        builder.progress.connect(lambda i, n: self.progress.emit(i, n))

        builder.set_temperature_history(t, T)

        for pulse in config["/multiple_pulse/pulses"]:
            t1 = units.Q_(pulse["arrival_time"]).to("s")
            t2 = t1 + units.Q_(pulse["duration"]).to("s")
            scale = pulse["scale"]
            builder.add_contribution(t1.magnitude, scale)
            builder.add_contribution(t2.magnitude, -scale)

        self.status.emit("Building temperature history")
        Tmp = builder.build()

        self.status.emit("Writing temperature history")

        data[:, 1] = Tmp

        output_paths["output_config_file_path"].write_text(yaml.dump(config.tree))
        fmt = config["/multiple_pulse/output_file_format"]
        if fmt is None:
            fmt = output_paths["output_file_path"].suffix[1:]
        if fmt is None:
            fmt = "txt"

        utils.write_to_file(output_paths["output_file_path"], data, fmt)
        self.status.emit("done")


@app.command()
def multiple_pulse(
    config_file: Path,
    njobs: Annotated[int, typer.Option(help="Number of parallel jobs to run.")] = None,
    skip_existing_outputs: Annotated[
        bool,
        typer.Option(
            help="Don't run simulation if the output file that would be written already exists."
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option(help="Print extra information")] = False,
    quiet: Annotated[bool, typer.Option(help="Don't print to console.")] = False,
):
    iconsole = rich.console.Console(stderr=False, quiet=quiet)
    vconsole = rich.console.Console(
        stderr=False, quiet=True if quiet or not verbose else False
    )
    econsole = rich.console.Console(stderr=True)

    iconsole.print("Loading configuration(s)")
    try:
        # we need to convert all quantities to strings because they will ahve been created
        # with a different unit registry than the models are using.
        configs = powerconf.yaml.powerload(
            config_file, njobs=multiprocessing.cpu_count(), transform=q2str
        )
    except KeyError as e:
        econsole.print(
            "[red]A configuration parameter references another non-existent parameter.[/red]"
        )

        econsole.print("\n\n[red]" + str(e) + "[/red]\n\n")
        raise typer.Exit(1)
    iconsole.print(f"done. Loaded {len(configs)} configurations.")

    # validate configs
    iconsole.print("Validating configuration(s)")
    for i, config in enumerate(configs):
        try:
            config = MultiplePulseCmdConfig(**config.tree)
            configs[i] = fspathtree(config.model_dump())
        except ValidationError as e:
            econsole.print(
                "[red]There was an error reading the configuration file.[/red]"
            )
            econsole.print("\n\nPydantic Error Message:")
            econsole.print(e)
            econsole.print("\n\n")
            raise typer.Exit(1)
    iconsole.print("done")

    if len(configs) > 1:
        # disable printing status information when we are processing multiple configurations
        console.print = lambda *args, **kwargs: None

    if skip_existing_outputs:
        for c in configs:
            c["/skip_existing_outputs"] = True

    if njobs is None:
        njobs = min(multiprocessing.cpu_count(), len(configs))

    iconsole.print("Setting up parallel job processor and running configs")
    controller = parallel_jobs.BatchJobController(MultiplePulseProcess, njobs=njobs)
    controller.start()

    progress_display = (
        parallel_jobs.SilentProgressDisplay()
        if quiet
        else parallel_jobs.ProgressDisplay()
    )
    progress_display.setup_new_bar("Total")
    progress_display.set_total("Total", len(configs))
    for i in range(njobs):
        progress_display.setup_new_bar(f"Job-{i:03}")
    for i in range(njobs):
        progress_display.set_progress(f"Job-{i:03}", 0, 1)

    controller.progress.connect(
        lambda proc, prog: progress_display.set_progress(f"Job-{proc:03}", *prog)
    )
    controller.status.connect(
        lambda proc, msg: (
            progress_display.set_progress(f"Job-{proc:03}", 1, 1)
            if msg == "done"
            else None
        )
    )
    controller.status.connect(
        lambda proc, msg: (
            progress_display.update_progress("Total") if msg == "done" else None
        )
    )

    iconsole.print("Running jobs")
    controller.run_jobs(configs)
    controller.stop()
    controller.wait()

    raise typer.Exit(0)


class DamageConfig(config.BaseModel):
    input_file: Path
    output_config_file: Path
    output_file: Path

    A: config.QuantityWithUnit("1/s")
    Ea: config.QuantityWithUnit("J/mol")
    T0: config.QuantityWithUnit("K")


class DamageCmdConfig(config.BaseModel):
    damage: DamageConfig


@app.command()
def damage(
    config_file: Path,
    njobs: Annotated[str, typer.Option(help="Number of parallel jobs to run.")] = None,
    skip_existing_outputs: Annotated[
        bool,
        typer.Option(
            help="Don't compute damage for temperature profile if the output file that would be written already exists."
        ),
    ] = False,
    write_threshold_profiles: Annotated[
        bool,
        typer.Option(
            help="After computing the damage threshold scaling factor, write a scaled Tvst file that corresponds to the damage threshold temperature profile."
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option(help="Print extra information")] = False,
    quiet: Annotated[bool, typer.Option(help="Don't print to console.")] = False,
):
    iconsole = rich.console.Console(stderr=False, quiet=quiet)
    vconsole = rich.console.Console(
        stderr=False, quiet=True if quiet or not verbose else False
    )
    econsole = rich.console.Console(stderr=True)
    arrhenius_cli_exec = shutil.which("Arrhenius-cli")
    if arrhenius_cli_exec is None:
        econsole.print("[red]`Arrhenius-cli` not found.[/red]")
        econsole.print(
            "`retina-therm damage` uses `Arrhenius-cli` to compute damage thresholds. Please install it."
        )
        econsole.print("see https://github.com/CD3/libArrhenius")
        raise typer.Exit(1)

    try:
        # we need to convert all quantities to strings because they will ahve been created
        # with a different unit registry than the models are using.
        configs = powerconf.yaml.powerload(
            config_file, njobs=multiprocessing.cpu_count(), transform=q2str
        )
    except KeyError as e:
        econsole.print(
            "[red]A configuration parameter references another non-existent parameter.[/red]"
        )

        econsole.print("\n\n[red]" + str(e) + "[/red]\n\n")
        raise typer.Exit(1)

    # validate configs
    for i, config in enumerate(configs):
        try:
            # validate the input config using pydantic model
            config = DamageCmdConfig(**config.tree)
            # copy model back to config so that default values will be filled in
            configs[i] = fspathtree(config.model_dump())

        except ValidationError as e:
            econsole.print(
                "[red]There was an error reading the configuration file.[/red]"
            )
            econsole.print("\n\nPydantic Error Message:")
            econsole.print(e)
            econsole.print("\n\n")
            raise typer.Exit(1)

    cmds = []
    for config in configs:
        output_file = config["/damage/output_file"]
        if skip_existing_outputs and output_file.exists():
            console.print(f"Output file `{output_file}` already exists. Skipping.")
            continue
        output_config_file = config["/damage/output_config_file"]

        Tvst_file = config["/damage/input_file"]
        A = units.Q_(config["/damage/A"]).magnitude
        Ea = units.Q_(config["/damage/Ea"]).magnitude
        T0 = units.Q_(config["/damage/T0"]).magnitude
        cmd = f"Arrhenius-cli calc-threshold '{Tvst_file}' --A {A} --Ea {Ea} --T0 {T0}"
        if write_threshold_profiles:
            cmd += " --write-threshold-profiles"
        iconsole.print(f"Running `{cmd}`")
        output = subprocess.check_output(cmd, shell=True).decode()
        scale = output.split("\n")[1].split("|")[-1].strip()

        output_file.write_text(f"scale: {scale}\n")
        output_config_file.write_text(yaml.dump(config.tree))

    print()


class TruncateTemperatureProfileProcess(parallel_jobs.JobProcessorBase):
    def run_job(self, config):
        file = config["file"]
        threshold = config["threshold"]

        self.status.emit(f"Truncating temperature_history in {file}.")

        self.progress.emit(0, 4)
        data = numpy.loadtxt(file)
        data = utils.read_from_file(
            file, config.get("file_format", Path(file).suffix[1:])
        )
        self.progress.emit(1, 4)
        threshold = units.Q_(threshold)
        if threshold.check(""):
            Tmax = max(data[:, 1])
            Tthreshold = threshold.magnitude * Tmax
        elif threshold.check("K"):
            Tthreshold = threshold.magnitude

        if data[-1, 1] > Tthreshold:
            self.status.emit(f"{file} already trucated...skipping.")
            self.progress.emit(4, 4)
            return

        self.progress(2, 4)
        idx = numpy.argmax(numpy.flip(data[:, 1]) > Tthreshold)
        self.progress(3, 4)
        self.status.emit(f"Saving trucated history back to {file}.")
        numpy.savetxt(file, data[:-idx, :])
        self.progress.emit(4, 4)
        self.status.emit(f"done")


@app.command()
def truncate_temperature_history_file(
    temperature_history_file: List[Path],
    njobs: Annotated[str, typer.Option(help="Number of parallel jobs to run.")] = None,
    threshold: Annotated[
        str,
        typer.Option(
            help="Threshold temperature for truncating. Can be a temperature or a fraction. If a fraction is given, the threshold temperature will be computed as threshold*Tmax."
        ),
    ] = "0.001",
):
    """
    Truncate a temperature history file, removing all point in the end of the history where the temperature is below threshold*Tmax.
    This is used to decrease the size of the temperature history so that computing damage thresholds is faster.
    """
    threshold = units.Q_(threshold)
    if not threshold.check("") and not threshold.check("K"):
        raise typer.Exit(f"threshold must be a temperature or dimensionless")

    configs = []
    for file in temperature_history_file:
        configs.append({"file": file, "threshold": threshold})

    if njobs is None:
        njobs = multiprocessing.cpu_count()

    controller = parallel_jobs.BatchJobController(
        TruncateTemperatureProfileProcess, njobs=njobs
    )

    controller.status.connect(lambda proc, msg: print(msg))
    controller.start()
    controller.run_jobs(configs)
    controller.stop()
    controller.wait()


# _            _                           _     _
# | |_ ___   __| | ___ _   _ __   ___  _ __| |_  | |_ ___    _ __   _____      __
# | __/ _ \ / _` |/ _ (_) | '_ \ / _ \| '__| __| | __/ _ \  | '_ \ / _ \ \ /\ / /
# | || (_) | (_| | (_) |  | |_) | (_) | |  | |_  | || (_) | | | | |  __/\ V  V /
# \__\___/ \__,_|\___(_) | .__/ \___/|_|   \__|  \__\___/  |_| |_|\___| \_/\_/
#                        |_|
#                  __ _
#  ___ ___  _ __  / _(_) __ _
# / __/ _ \| '_ \| |_| |/ _` |
# | (_| (_) | | | |  _| | (_| |
# \___\___/|_| |_|_| |_|\__, |
#                       |___/
#  __                                             _
# / _|_ __ __ _ _ __ ___   _____      _____  _ __| | __
# | |_| '__/ _` | '_ ` _ \ / _ \ \ /\ / / _ \| '__| |/ /
# |  _| | | (_| | | | | | |  __/\ V  V / (_) | |  |   <
# |_| |_|  \__,_|_| |_| |_|\___| \_/\_/ \___/|_|  |_|\_\


# @app.command()
# def print_config_ids(
#    config_file: Path,
# ):
#    """Print IDs of configuration in CONFIG_FILES. Useful for determining if a configuration has already been ran."""
#    configs = powerconf.yaml.powerload(config_file)
#    configs = powerconf.utils.apply_transform(
#        configs, lambda p, n: str(n), predicate=lambda p, n: hasattr(n, "magnitude")
#    )
#    configs = list(map(lambda c: compute_tissue_properties(c), configs))
#    config_ids = list(map(powerconf.utils.get_id, configs))
#    for _id in config_ids:
#        print(_id)


# @app.command()
# def convert_file(
#    input_file: Path,
#    output_file: Path,
#    input_format: Annotated[
#        str, typer.Option("--input-format", "-f", help="Input file format")
#    ] = None,
#    output_format: Annotated[
#        str, typer.Option("--output-format", "-t", help="Output file format")
#    ] = None,
#    filetype: Annotated[str, typer.Option(help="File type (e.g. Tvst)")] = None,
# ):
#    if not input_file.exists():
#        print(f"ERROR: {input_file} does not exists.")
#        raise typer.Exit(1)

#    formats = ["txt", "hd5", "rt"]

#    if input_format is None:
#        input_format = input_file.suffix[1:]

#    if output_format is None:
#        output_format = output_file.suffix[1:]

#    print(f"{input_file}({input_format}) -> {output_file}({output_format})")

#    data = utils.read_Tvst_from_file(input_file, input_format)
#    data = utils.write_Tvst_to_file(data, output_file, output_format)


# class RelaxationTimeProcess(parallel_jobs.JobProcessorBase):
#    def run_job(self, config):
#        G = greens_functions.MultiLayerGreensFunction(config.tree)
#        threshold = config["relaxation_time/threshold"]
#        dt = config.get("simulation/time/dt", "1 us")
#        dt = units.Q_(dt).to("s").magnitude
#        tmax = config.get("simulation/time/max", "1 year")
#        tmax = units.Q_(tmax).to("s").magnitude
#        z = config.get("sensor/z", "0 um")
#        z = units.Q_(z).to("cm").magnitude
#        r = config.get("sensor/r", "0 um")
#        r = units.Q_(r).to("cm").magnitude
#        i = 0
#        t = i * dt
#        T = G(z, t)
#        Tp = T
#        Tth = threshold * Tp

#        status.emit(f"Looking for {threshold} thermal relaxation time.\n")
#        status.emit(f"Peak temperature is {mp.nstr(Tp, 5)}\n")
#        status.emit(f"Looking for time to {mp.nstr(Tth, 5)}\n")
#        i = 1
#        while T > threshold * Tp:
#            i *= 2
#            t = i * dt
#            T = G(z, t)
#        i_max = i
#        i_min = i / 2
#        status.emit(f"Relaxation time bracketed: [{i_min*dt},{i_max*dt}]\n")

#        t = utils.bisect(lambda t: G(z, r, t) - Tth, i_min * dt, i_max * dt)
#        t = sum(t) / 2
#        T = G(z, r, t)

#        status.emit(f"time: {mp.nstr(mp.mpf(t), 5)}\n")
#        status.emit(f"Temperature: {mp.nstr(T, 5)}\n")


# @app.command()
# def relaxation_time(
#    config_file: Path,
#    dps: Annotated[
#        int,
#        typer.Option(help="The precision to use for calculations when mpmath is used."),
#    ] = 100,
#    threshold: Annotated[float, typer.Option()] = 0.01,
# ):
#    configs = load_config(config_file, override)

#    mp.dps = dps

#    jobs = []
#    # create the jobs to run
#    for config in configs:
#        config["relaxation_time/threshold"] = threshold
#        jobs.append(multiprocessing.Process(target=relaxation_time_job, args=(config,)))
#    # run the jobs
#    for job in jobs:
#        job.start()
#    # wait on the jobs
#    for job in jobs:
#        job.join()


# class ImpulseResponseProcess(parallel_jobs.JobProcessorBase):
#    def run_job(self, config):
#        config_id = powerconf.utils.get_id(config)

#        G = greens_functions.MultiLayerGreensFunction(config.tree)
#        eval_times = compute_evaluation_times(config)
#        z = config.get("sensor/z", "0 um")
#        z = units.Q_(z).to("cm").magnitude
#        r = config.get("sensor/r", "0 um")
#        r = units.Q_(r).to("cm").magnitude

#        ctx = {
#            "config_id": config_id,
#            "c": config,
#        }

#        output_paths = {}
#        for k in ["simulation/output_file", "simulation/output_config_file"]:
#            filename = config.get(k, None)
#            output_paths[k + "_path"] = Path("/dev/stdout")
#            if filename is not None:
#                try:
#                    filename = filename.format(**ctx).replace(" ", "_")
#                except:
#                    raise RuntimeError(
#                        f"There was an error trying to generate output filename from template '{filename}'."
#                    )
#                path = Path(filename)
#                output_paths[k + "_path"] = path
#                if path.parent != Path():
#                    path.parent.mkdir(parents=True, exist_ok=True)

#        output_paths["simulation/output_config_file_path"].write_text(
#            yaml.dump(config.tree)
#        )

#        with output_paths["simulation/output_file_path"].open("w") as f:
#            for t in eval_times:
#                T = G(z, r, t)
#                f.write(f"{t} {T}\n")

#        self.status.emit("done.")


# @app.command()
# def impulse_response(
#    config_file: Path,
#    jobs: int = None,
#    dps: Annotated[
#        int,
#        typer.Option(help="The precision to use for calculations when mpmath is used."),
#    ] = 100,
# ):
#    mp.dps = dps

#    configs = powerconf.yaml.powerload(config_file)
#    configs = powerconf.utils.apply_transform(
#        configs, lambda p, n: str(n), predicate=lambda p, n: hasattr(n, "magnitude")
#    )
#    for config in configs:
#        if "/impulse_response/threshold" not in config:
#            config["/impulse_response/threshold"] = 0.01

#    if jobs is None or jobs > 1:
#        jobs = min(multiprocessing.cpu_count(), len(configs_to_run))
#        controller = parallel_jobs.Controller(ImpulseResponseProcess, jobs)
#        controller.run(configs)
#        controller.stop()
#    else:
#        p = ImpulseResponseProcess()
#        for config in configs:
#            p.run_job(config)

#    raise typer.Exit(0)


# @app.command()
# def config(
#    print_multiple_pulse_example_config: Annotated[
#        bool,
#        typer.Option(
#            help="Print an example configuration file for the multiple-pulse command and exit."
#        ),
#    ] = False,
#    print_temperature_rise_example_config: Annotated[
#        bool,
#        typer.Option(
#            help="Print an example configuration file for the temperature-rise command and exit."
#        ),
#    ] = False,
# ):
#    """Various config file related task. i.e. print example config, etc."""
#    print("Under Developement")
#    return

#    if print_multiple_pulse_example_config:
#        config = fspathtree()
#        config["/input_file"] = "input/CW/Tvst.txt"
#        config["/output_file"] = "output/MP/{c[tau]}-{c[N]}-Tvst.txt"
#        config["/output_config_file"] = "output/MP/{c[tau]}-{c[N]}-CONFIG.yml"
#        config["/tau"] = "100 us"
#        config["/t0"] = "100 us"
#        config["/N"] = 100
#        print(yaml.dump(config.tree))
#        raise typer.Exit(1)

#    if print_temperature_rise_example_config:
#        config = fspathtree()
#        config["/thermal/k"] = "0.6306 W/m/K"
#        config["/thermal/rho"] = "992 kg/m^3"
#        config["/thermal/c"] = "4178 J /kg / K"
#        config["/layers/0/name"] = "RPE"
#        config["/layers/0/z0"] = "0 um"
#        config["/layers/0/d"] = "10 um"
#        config["/layers/0/mua"] = "720 1/cm"
#        config["/layers/1/name"] = "Choroid"
#        config["/layers/1/z0"] = "4 um"
#        config["/layers/1/d"] = "20 um"
#        config["/layers/1/mua"] = "140 1/cm"
#        config["/laser/E0"] = "1 W/cm^2"
#        config["/laser/D"] = "100 um"
#        config["/laser/profile"] = "flattop"
#        config["/sensor/z"] = "0 um"
#        config["/sensor/r"] = "0 um"
#        config["/temperature_rise/use_approximations"] = True
#        config["/temperature_rise/temperature_rise/method"] = "quad"
#        config["/temperature_rise/output_file"] = (
#            "output/CW/{c[/laser/D]}-{c[/sensor/r]}-Tvst.txt"
#        )
#        config["/temperature_rise/output_config_file"] = (
#            "output/CW/{c[/laser/D]}-{c[/sensor/r]}-CONFIG.yml"
#        )
#        config["/temperature_rise/time/dt"] = "1 us"
#        config["/temperature_rise/time/max"] = "10 ms"

#        print(yaml.dump(config.tree))
#        raise typer.Exit(1)


# @app.command()
# def multipulse_microcavitation_threshold(
#     config_file: Path,
#     dps: Annotated[
#         int,
#         typer.Option(help="The precision to use for calculations when mpmath is used."),
#     ] = 100,
#     override: Annotated[
#         list[str],
#         typer.Option(
#             help="key=val string to override a configuration parameter. i.e. --parameter 'simulation/time/dt=2 us'"
#         ),
#     ] = [],
# ):
#     configs = load_config(config_file, override)
#     mp.dps = dps

#     for config in configs:
#         T0 = config.get("baseline_temperature", "37 degC")
#         toks = T0.split(maxsplit=1)
#         T0 = units.Q_(float(toks[0]), toks[1]).to("K")
#         Tnuc = config.get("microcavitation/Tnuc", "116 degC")
#         toks = Tnuc.split(maxsplit=1)
#         Tnuc = units.Q_(float(toks[0]), toks[1]).to("K")
#         m = units.Q_(config.get("microcavitation/m", "-1 mJ/cm^2/K"))
#         PRF = units.Q_(config.get("laser/PRF", "1 kHz"))
#         t0 = 1 / PRF
#         t0 = t0.to("s").magnitude
#         N = 1000

#         output_file = config.get("simulation/output_file", "Hth_vs_N.txt")

#         config["laser/E0"] = "1 W/cm^2"  # override user power
#         G = greens_functions.MultiLayerGreensFunction(config.tree)
#         z = config.get("sensor/z", "0 um")
#         z = units.Q_(z).to("cm").magnitude
#         r = config.get("sensor/r", "0 um")
#         r = units.Q_(r).to("cm").magnitude

#         T = numpy.zeros([N])

#         for i in range(1, len(T)):
#             T[i] = T[i - 1] + G(z, r, t0 * i)

#         with output_file.open("w") as f:
#             for n in range(1, N):
#                 H = (m * T0 - m * Tnuc) / (1 - m * units.Q_(T[n - 1], "K/(J/cm^2)"))
#                 f.write(f"{n} {H}\n")


@app.command()
def status(
    config_file: Path,
):

    iconsole = rich.console.Console(stderr=False)
    econsole = rich.console.Console(stderr=True)

    iconsole.print("Loading configuration(s)")
    try:
        configs = powerconf.yaml.powerload(
            config_file, njobs=multiprocessing.cpu_count()
        )
    except KeyError as e:
        econsole.print(
            "[red]A configuration parameter references another non-existent parameter.[/red]"
        )

        econsole.print("\n\n[red]" + str(e) + "[/red]\n\n")
        raise typer.Exit(1)
    iconsole.print("done.")

    output_files = {}
    output_files_exist = {}
    for config in configs:
        for node in config.get_all_leaf_node_paths():
            if node.name == "output_file":
                if node not in output_files:
                    output_files[node] = []
                output_files[node].append(Path(config[node]))
    for k in output_files:
        output_files_exist[k] = list(map(lambda p: p.exists(), output_files[k]))
    iconsole.print(
        f"{len(configs)} configuration instances producing {len(output_files)} output files."
    )

    finished = sum(itertools.chain(*output_files_exist.values()))
    total = len(list(itertools.chain(*output_files_exist.values())))

    iconsole.print(
        f"{ finished } ({ 100*finished/total}%) output files have already been generated."
    )
    for k in output_files_exist:
        iconsole.print(
            "    ", k, sum(output_files_exist[k]), "/", len(output_files_exist[k])
        )


@app.command()
def report(
    config_file: Path,
    output_file: Annotated[
        Path, typer.Argument(help="File to write report to.")
    ] = Path("/dev/stdout"),
    format: Annotated[
        str, typer.Argument(help="Report format. Currently only 'txt' is supported.")
    ] = "txt",
):
    iconsole = rich.console.Console(stderr=False)
    econsole = rich.console.Console(stderr=True)
    iconsole.print("Loading configuration(s)")
    try:
        configs = powerconf.yaml.powerload(
            config_file, njobs=multiprocessing.cpu_count()
        )
    except KeyError as e:
        econsole.print(
            "[red]A configuration parameter references another non-existent parameter.[/red]"
        )

        econsole.print("\n\n[red]" + str(e) + "[/red]\n\n")
        raise typer.Exit(1)
    iconsole.print("done.")

    rows = []
    rows.append(
        list(
            map(
                lambda n: (
                    n["title"]
                    if "unit" not in n
                    else n["title"] + " [" + n["unit"] + "]"
                ),
                configs[0]["/report/columns"],
            )
        )
    )
    for config in configs:
        row = []
        for i in range(len(rows[0])):
            v = config[f"/report/columns/{i}/value"]
            if v is None:
                v = "--"
            elif "unit" in config[f"/report/columns/{i}"]:
                v.ito(config[f"/report/columns/{i}/unit"])
                v = str(v.magnitude)
                print(v)
            else:
                v = str(v)
            row.append(v)
        rows.append(row)

    if format == "txt":
        with output_file.open("w") as f:
            for row in rows:
                f.write("|".join(row))
                f.write("\n")
