import multiprocessing
import pprint
import sys
from pathlib import Path
from typing import Annotated, List

import numpy
import powerconf
import rich
import scipy
import tissue_properties.optical.absorption_coefficient.schulmeister
import tissue_properties.optical.ocular_transmission.schulmeister
import typer
import yaml
from fspathtree import fspathtree
from mpmath import mp
from tqdm import tqdm

import retina_therm
from retina_therm import (greens_functions, multi_pulse_builder, schemas,
                          units, utils)

from . import config_utils, parallel_jobs, utils

app = typer.Typer()
console = rich.console.Console()


invoked_subcommand = None
config_filename_stem = None


class models:
    class schulmeister:
        class mua:
            RPE = tissue_properties.optical.absorption_coefficient.schulmeister.RPE()
            HenlesFiberLayer = (
                tissue_properties.optical.absorption_coefficient.schulmeister.HenlesFiberLayer()
            )
            Choroid = (
                tissue_properties.optical.absorption_coefficient.schulmeister.Choroid()
            )


@app.callback()
def main(ctx: typer.Context):
    global invoked_subcommand
    invoked_subcommand = ctx.invoked_subcommand


def compute_evaluation_times(config):
    # if times are given in the config, just them
    if "simulation/time/ts" in config:
        t = numpy.array(
            [units.Q_(time).to("s").magnitude for time in config["simulation/time/ts"]]
        )
    else:
        # we want to support specifying the times as a single range,
        # i.e. "from tmin to tmax by steps of dt"
        # or multiple ranges
        # i.e. "from tmin_1 to tmax_1 by steps of dt_1 AND from tmin_2 to tmax_2 by steps of dt_2"
        # this is usefull for sampling the start of a long exposure at higher resolution than the end.
        time_configs = []
        if type(config["simulation/time"].tree) == dict:
            time_configs.append(config["simulation/time"])
        else:
            for c in config["simulation/time"]:
                time_configs.append(c)

        time_arrays = []
        for i, time_config in enumerate(time_configs):
            dt = units.Q_(time_config.get("dt", "1 us"))
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


class RelaxationTimeProcess(parallel_jobs.JobProcess):
    def run_job(self, config):
        G = greens_functions.MultiLayerGreensFunction(config.tree)
        threshold = config["relaxation_time/threshold"]
        dt = config.get("simulation/time/dt", "1 us")
        dt = units.Q_(dt).to("s").magnitude
        tmax = config.get("simulation/time/max", "1 year")
        tmax = units.Q_(tmax).to("s").magnitude
        z = config.get("sensor/z", "0 um")
        z = units.Q_(z).to("cm").magnitude
        r = config.get("sensor/r", "0 um")
        r = units.Q_(r).to("cm").magnitude
        i = 0
        t = i * dt
        T = G(z, t)
        Tp = T
        Tth = threshold * Tp

        status.emit(f"Looking for {threshold} thermal relaxation time.\n")
        status.emit(f"Peak temperature is {mp.nstr(Tp, 5)}\n")
        status.emit(f"Looking for time to {mp.nstr(Tth, 5)}\n")
        i = 1
        while T > threshold * Tp:
            i *= 2
            t = i * dt
            T = G(z, t)
        i_max = i
        i_min = i / 2
        status.emit(f"Relaxation time bracketed: [{i_min*dt},{i_max*dt}]\n")

        t = utils.bisect(lambda t: G(z, r, t) - Tth, i_min * dt, i_max * dt)
        t = sum(t) / 2
        T = G(z, r, t)

        status.emit(f"time: {mp.nstr(mp.mpf(t), 5)}\n")
        status.emit(f"Temperature: {mp.nstr(T, 5)}\n")


@app.command()
def relaxation_time(
    config_file: Path,
    dps: Annotated[
        int,
        typer.Option(help="The precision to use for calculations when mpmath is used."),
    ] = 100,
    threshold: Annotated[float, typer.Option()] = 0.01,
):
    configs = load_config(config_file, override)

    mp.dps = dps

    jobs = []
    # create the jobs to run
    for config in configs:
        config["relaxation_time/threshold"] = threshold
        jobs.append(multiprocessing.Process(target=relaxation_time_job, args=(config,)))
    # run the jobs
    for job in jobs:
        job.start()
    # wait on the jobs
    for job in jobs:
        job.join()


class ImpulseResponseProcess(parallel_jobs.JobProcess):
    def run_job(self, config):
        config_id = powerconf.utils.get_id(config)

        G = greens_functions.MultiLayerGreensFunction(config.tree)
        eval_times = compute_evaluation_times(config)
        z = config.get("sensor/z", "0 um")
        z = units.Q_(z).to("cm").magnitude
        r = config.get("sensor/r", "0 um")
        r = units.Q_(r).to("cm").magnitude

        ctx = {
            "config_id": config_id,
            "c": config,
        }

        output_paths = {}
        for k in ["simulation/output_file", "simulation/output_config_file"]:
            filename = config.get(k, None)
            output_paths[k + "_path"] = Path("/dev/stdout")
            if filename is not None:
                try:
                    filename = filename.format(**ctx).replace(" ", "_")
                except:
                    raise RuntimeError(
                        f"There was an error trying to generate output filename from template '{filename}'."
                    )
                path = Path(filename)
                output_paths[k + "_path"] = path
                if path.parent != Path():
                    path.parent.mkdir(parents=True, exist_ok=True)

        output_paths["simulation/output_config_file_path"].write_text(
            yaml.dump(config.tree)
        )

        with output_paths["simulation/output_file_path"].open("w") as f:
            for t in eval_times:
                T = G(z, r, t)
                f.write(f"{t} {T}\n")

        self.status.emit("done.")


@app.command()
def impulse_response(
    config_file: Path,
    jobs: int = None,
    dps: Annotated[
        int,
        typer.Option(help="The precision to use for calculations when mpmath is used."),
    ] = 100,
):
    mp.dps = dps

    configs = powerconf.yaml.powerload(config_file)
    configs = powerconf.utils.apply_transform(
        configs, lambda p, n: str(n), predicate=lambda p, n: hasattr(n, "magnitude")
    )
    for config in configs:
        if "/impulse_response/threshold" not in config:
            config["/impulse_response/threshold"] = 0.01

    if jobs is None or jobs > 1:
        controller = parallel_jobs.Controller(ImpulseResponseProcess, jobs)
        controller.run(configs)
        controller.stop()
    else:
        p = ImpulseResponseProcess()
        for config in configs:
            p.run_job(config)

    raise typer.Exit(0)


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
            layer["mua"] = str(mua)  # schema validators expect strings for quantities
    return config


class TemperatureRiseProcess(parallel_jobs.JobProcess):
    def run_job(self, config):
        config_id = powerconf.utils.get_id(config)
        # computing tissue properties in main process now so
        # we can compute config ID with expanded values
        # config = compute_tissue_properties(config)

        if "laser/profile" not in config:
            config["laser/profile"] = "flattop"
        if "simulation/with_units" not in config:
            config["simulation/with_units"] = False
        if "simulation/use_approximations" not in config:
            config["simulation/with_approximations"] = False
        if "simulation/use_multi_precision" not in config:
            config["simulation/with_multi_precision"] = False

        if "laser/pulse_duration" not in config:
            G = greens_functions.CWRetinaLaserExposure(config.tree)
        else:
            G = greens_functions.PulsedRetinaLaserExposure(config.tree)
        z = config.get("sensor/z", "0 um")
        z = units.Q_(z).to("cm").magnitude
        r = config.get("sensor/r", "0 um")
        r = units.Q_(r).to("cm").magnitude

        # if times are given in the config, just them
        t = compute_evaluation_times(config)
        method = config.get("simulation/temperature_rise/method", "quad")
        if method not in temperature_rise_integration_methods + ["undefined"]:
            raise RuntimeError(f"Unrecognized integration method '{method}'")

        self.status.emit("Computing temperature rise...")
        G.progress.connect(lambda i, n: self.progress.emit(i, n))
        T = G.temperature_rise(z, r, t, method=method)
        self.status.emit("done.")
        self.status.emit("Writing output files...")
        ctx = {
            "config_id": config_id,
            "c": config,
        }

        output_paths = {}
        for k in ["simulation/output_file", "simulation/output_config_file"]:
            filename = config.get(k, None)
            output_paths[k + "_path"] = Path("/dev/stdout")
            if filename is not None:
                try:
                    filename = filename.format(**ctx).replace(" ", "_")
                except:
                    raise RuntimeError(
                        f"There was an error trying to generate output filename from template '{filename}'."
                    )
                path = Path(filename)
                output_paths[k + "_path"] = path
                if path.parent != Path():
                    path.parent.mkdir(parents=True, exist_ok=True)

        output_paths["simulation/output_config_file_path"].write_text(
            yaml.dump(config.tree)
        )
        utils.write_to_file(
            output_paths["simulation/output_file_path"],
            numpy.c_[t, T],
            config.get(
                "simulation/output_format",
                output_paths["simulation/output_file_path"].suffix[1:],
            ),
        )
        self.status.emit("done.")


@app.command()
def temperature_rise(
    config_file: Path,
    jobs: Annotated[int, typer.Option(help="Number of parallel jobs to run.")] = None,
    ids: Annotated[
        List[str],
        typer.Option(
            help="Only run simulation for configurations with ID in the given list."
        ),
    ] = [],
    dps: Annotated[
        int,
        typer.Option(help="The precision to use for calculations when mpmath is used."),
    ] = 100,
    list_methods: Annotated[
        bool, typer.Option(help="List the avaiable integration methods.")
    ] = False,
    print_ids: Annotated[
        bool,
        typer.Option(
            help="Load configuration file(s) and print a list of the config IDs."
        ),
    ] = False,
):
    if list_methods:
        print("Available inegration methods:")
        for m in temperature_rise_integration_methods:
            print("  ", m)
        raise typer.Exit(0)

    mp.dps = dps

    configs = powerconf.yaml.powerload(config_file)
    # we need to convert all quantities to strings before we pass
    # them to the implementation classes. they do validation based on
    # string representations
    configs = powerconf.utils.apply_transform(
        configs, lambda p, n: str(n), predicate=lambda p, n: hasattr(n, "magnitude")
    )
    configs = list(map(lambda c: compute_tissue_properties(c), configs))

    config_ids = list(map(powerconf.utils.get_id, configs))
    if print_ids:
        for _id in config_ids:
            print(_id)
        raise typer.Exit(0)

    if len(ids) == 0:
        ids = config_ids

    configs_to_run = list(filter(lambda c: powerconf.utils.get_id(c) in ids, configs))
    if len(configs_to_run) == 0:
        rich.print("[orange]No configurations matched list of IDs to run[/orange]")
    if len(configs_to_run) > 1:
        # disable printing status information when we are processing multiple configurations
        console.print = lambda *args, **kwargs: None

    if jobs is None or jobs > 1:
        controller = parallel_jobs.Controller(TemperatureRiseProcess, jobs)
        controller.run(configs_to_run)
        controller.stop()
    else:
        p = TemperatureRiseProcess()
        for config in configs_to_run:
            p.run_job(config)

    raise typer.Exit(0)


@app.command()
def multipulse_microcavitation_threshold(
    config_file: Path,
    dps: Annotated[
        int,
        typer.Option(help="The precision to use for calculations when mpmath is used."),
    ] = 100,
    override: Annotated[
        list[str],
        typer.Option(
            help="key=val string to override a configuration parameter. i.e. --parameter 'simulation/time/dt=2 us'"
        ),
    ] = [],
):
    configs = load_config(config_file, override)
    mp.dps = dps

    for config in configs:
        T0 = config.get("baseline_temperature", "37 degC")
        toks = T0.split(maxsplit=1)
        T0 = units.Q_(float(toks[0]), toks[1]).to("K")
        Tnuc = config.get("microcavitation/Tnuc", "116 degC")
        toks = Tnuc.split(maxsplit=1)
        Tnuc = units.Q_(float(toks[0]), toks[1]).to("K")
        m = units.Q_(config.get("microcavitation/m", "-1 mJ/cm^2/K"))
        PRF = units.Q_(config.get("laser/PRF", "1 kHz"))
        t0 = 1 / PRF
        t0 = t0.to("s").magnitude
        N = 1000

        output_file = config.get("simulation/output_file", "Hth_vs_N.txt")

        config["laser/E0"] = "1 W/cm^2"  # override user power
        G = greens_functions.MultiLayerGreensFunction(config.tree)
        z = config.get("sensor/z", "0 um")
        z = units.Q_(z).to("cm").magnitude
        r = config.get("sensor/r", "0 um")
        r = units.Q_(r).to("cm").magnitude

        T = numpy.zeros([N])

        for i in range(1, len(T)):
            T[i] = T[i - 1] + G(z, r, t0 * i)

        with output_file.open("w") as f:
            for n in range(1, N):
                H = (m * T0 - m * Tnuc) / (1 - m * units.Q_(T[n - 1], "K/(J/cm^2)"))
                f.write(f"{n} {H}\n")


class MultiplePulseProcess(parallel_jobs.JobProcess):
    def run_job(self, config):
        config_id = powerconf.utils.get_id(config)
        self.status.emit(
            "Loading base temperature history for building multiple-pulse history."
        )
        input_file = Path(config["input_file"])
        data = utils.read_from_file(
            input_file, config.get("input_file_format", input_file.suffix[1:])
        )
        imax = len(data)
        tmax = units.Q_(data[-1, 0], "s")
        # if tmax is given in the config file, we want to trucate
        # the input data to include the first time >= tmax
        # this is an optimization reduces the size of the array we
        # are working.
        if "/tmax" in config:
            tmax = units.Q_(config["/tmax"])
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

        if not multi_pulse_builder.is_uniform_spaced(t):
            tp = multi_pulse_builder.regularize_grid(t)
            Tp = multi_pulse_builder.interpolate_temperature_history(t, T, tp)
            t = tp
            T = Tp
            data = numpy.zeros([len(tp), 2])
            data[:, 0] = t

        builder = multi_pulse_builder.MultiPulseBuilder()
        builder.progress.connect(lambda i, n: self.progress.emit(i, n))

        builder.set_temperature_history(t, T)

        # if a pulse duration is given, then it means we have a CW exposure
        # and we need to create the single pulse exposure by adding a -1 scale
        if "/tau" in config:
            builder.add_contribution(0, 1)
            builder.add_contribution(units.Q_(config["tau"]).to("s").magnitude, -1)
            Tsp = builder.build()
            builder.set_temperature_history(t, Tsp)
            builder.clear_contributions()

        contributions = []
        if "/contributions" in config:
            for c in config["/contributions"]:
                contributions.append(
                    {
                        "arrival_time": units.Q_(c["arrival_time"]).to("s").magnitude,
                        "scale": float(c["scale"]),
                    }
                )

        if (
            "/tau" in config
            and "/T" in config
            and "/N" in config
            and "/t0" not in config
        ):
            # if the total exposure time and number of pulses are given, we want
            # to compute the inter-pulse spacing that would fit the begining of
            # the first pulse and the end of the last pulse.
            tau = units.Q_(config["/tau"])
            T = units.Q_(config["/T"])
            N = units.Q_(config["/N"])
            if N < 2:
                config["/t0"] = str(tmax)

            else:
                t0 = (T - tau) / (N - 1)
                config["/t0"] = f"{t0:~}"

        if "/t0" in config:
            t0 = units.Q_(config["/t0"])
            if "/N" not in config:
                N = int((tmax / t0).magnitude)
                config["/N"] = f"{N:.2f}"

            N = units.Q_(config["/N"])

            arrival_time = units.Q_(0, "s")
            n = 0
            while arrival_time.to("s").magnitude < t[-1] and n < N:
                contributions.append(
                    {
                        "arrival_time": arrival_time.to("s").magnitude,
                        "scale": config.get("/scale", 1),
                    }
                )
                arrival_time += t0
                n += 1

        for c in contributions:
            builder.add_contribution(c["arrival_time"], c["scale"])

        self.status.emit("Building temperature history")
        Tmp = builder.build()

        self.status.emit("Writing temperature history")

        data[:, 1] = Tmp

        ctx = {
            "config_id": config_id,
            "c": config,
        }

        output_paths = {}
        for k in ["output_file", "output_config_file"]:
            filename = config.get(k, None)
            output_paths[k + "_path"] = Path("/dev/stdout")
            if filename is not None:
                filename = filename.format(**ctx).replace(" ", "_")
                path = Path(filename)
                output_paths[k + "_path"] = path
                if path.parent != Path():
                    path.parent.mkdir(parents=True, exist_ok=True)

        output_paths["output_config_file_path"].write_text(yaml.dump(config.tree))
        utils.write_to_file(
            output_paths["output_file_path"],
            data,
            config.get(
                "output_file_format", output_paths["output_file_path"].suffix[1:]
            ),
        )
        self.status.emit("done.")


@app.command()
def multiple_pulse(
    config_file: Path,
    jobs: Annotated[int, typer.Option(help="Number of parallel jobs to run.")] = None,
    ids: Annotated[
        List[str],
        typer.Option(
            help="Only run simulation for configurations with ID in the given list."
        ),
    ] = [],
    print_ids: Annotated[
        bool,
        typer.Option(
            help="Load configuration file(s) and print a list of the config IDs."
        ),
    ] = False,
):
    configs = powerconf.yaml.powerload(config_file)
    configs = list(
        filter(lambda c: "/remove" not in c or not any(c["/remove"]), configs)
    )
    configs = powerconf.utils.apply_transform(
        configs, lambda p, n: str(n), predicate=lambda p, n: hasattr(n, "magnitude")
    )
    # validate configs with pydantic
    # for config in configs:
    #     schemas.MultiplePulseCmdConfig(**config.tree)

    config_ids = list(map(powerconf.utils.get_id, configs))
    if print_ids:
        for _id in config_ids:
            print(_id)
        raise typer.Exit(0)

    if len(ids) == 0:
        ids = config_ids

    configs_to_run = list(filter(lambda c: powerconf.utils.get_id(c) in ids, configs))
    if len(configs_to_run) > 1:
        # disable printing status information when we are processing multiple configurations
        console.print = lambda *args, **kwargs: None

    if jobs is None or jobs > 1:
        controller = parallel_jobs.Controller(MultiplePulseProcess, jobs)
        controller.run(configs_to_run)
        controller.stop()
    else:
        p = MultiplePulseProcess()
        with tqdm(total=len(configs_to_run)) as pbar:
            for config in configs_to_run:
                try:
                    p.run_job(config)
                except Exception as e:
                    print("An exception was thrown rile running config")
                    print(config.tree)
                    raise e
                pbar.update(1)

    raise typer.Exit(0)


class TruncateTemperatureProfileProcess(parallel_jobs.JobProcess):
    def run_job(self, config):
        file = config["file"]
        threshold = config["threshold"]

        self.status(f"Truncating temperature_history in {file}.")
        self.progress(0, 4)
        data = numpy.loadtxt(file)
        data = utils.read_from_file(
            file, config.get("file_format", pathlib.Path(file).suffix[1:])
        )
        self.progress(1, 4)
        threshold = units.Q_(threshold)
        if threshold.check(""):
            Tmax = max(data[:, 1])
            Tthreshold = threshold.magnitude * Tmax
        elif threshold.check("K"):
            Tthreshold = threshold.magnitude

        if data[-1, 1] > Tthreshold:
            self.status(f"{file} already trucated...skipping.")
            self.progress(4, 4)
            return

        self.progress(2, 4)
        idx = numpy.argmax(numpy.flip(data[:, 1]) > Tthreshold)
        self.progress(3, 4)
        self.status(f"Saving trucated history back to {file}.")
        numpy.savetxt(file, data[:-idx, :])
        self.progress(4, 4)


@app.command()
def truncate_temperature_history_file(
    temperature_history_file: List[Path],
    threshold: Annotated[
        str,
        typer.Option(
            help="Threshold temperature for truncating. Can be a temperature or a fraction. If a fraction is given, the threshold temperature will be computed as threshold*Tmax."
        ),
    ] = "0.001",
    # line_count: Annotated[
    #     float,
    #     typer.Option(help="Line count for truncating."),
    # ],
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

    controller = parallel_jobs.Controller(TruncateTemperatureProfileProcess)
    controller.run(configs)
    controller.stop()


@app.command()
def print_config_ids(
    config_file: Path,
):
    """Print IDs of configuration in CONFIG_FILES. Useful for determining if a configuration has already been ran."""
    configs = powerconf.yaml.powerload(config_file)
    configs = powerconf.utils.apply_transform(
        configs, lambda p, n: str(n), predicate=lambda p, n: hasattr(n, "magnitude")
    )
    configs = list(map(lambda c: compute_tissue_properties(c), configs))
    config_ids = list(map(powerconf.utils.get_id, configs))
    for _id in config_ids:
        print(_id)


@app.command()
def config(
    print_multiple_pulse_example_config: Annotated[
        bool,
        typer.Option(
            help="Print an example configuration file for the multiple-pulse command and exit."
        ),
    ] = False,
    print_temperature_rise_example_config: Annotated[
        bool,
        typer.Option(
            help="Print an example configuration file for the temperature-rise command and exit."
        ),
    ] = False,
):
    """Various config file related task. i.e. print example config, etc."""

    if print_multiple_pulse_example_config:
        config = fspathtree()
        config["/input_file"] = "input/CW/Tvst.txt"
        config["/output_file"] = "output/MP/{c[tau]}-{c[N]}-Tvst.txt"
        config["/output_config_file"] = "output/MP/{c[tau]}-{c[N]}-CONFIG.yml"
        config["/tau"] = "100 us"
        config["/t0"] = "100 us"
        config["/N"] = 100
        print(yaml.dump(config.tree))
        raise typer.Exit(1)

    if print_temperature_rise_example_config:
        config = fspathtree()
        config["/thermal/k"] = "0.6306 W/m/K"
        config["/thermal/rho"] = "992 kg/m^3"
        config["/thermal/c"] = "4178 J /kg / K"
        config["/layers/0/name"] = "RPE"
        config["/layers/0/z0"] = "0 um"
        config["/layers/0/d"] = "10 um"
        config["/layers/0/mua"] = "720 1/cm"
        config["/layers/1/name"] = "Choroid"
        config["/layers/1/z0"] = "4 um"
        config["/layers/1/d"] = "20 um"
        config["/layers/1/mua"] = "140 1/cm"
        config["/laser/E0"] = "1 W/cm^2"
        config["/laser/D"] = "100 um"
        config["/laser/profile"] = "flattop"
        config["/sensor/z"] = "0 um"
        config["/sensor/r"] = "0 um"
        config["/simulation/use_approximations"] = True
        config["/simulation/temperature_rise/method"] = "quad"
        config["/simulation/output_file"] = (
            "output/CW/{c[/laser/D]}-{c[/sensor/r]}-Tvst.txt"
        )
        config["/simulation/output_config_file"] = (
            "output/CW/{c[/laser/D]}-{c[/sensor/r]}-CONFIG.yml"
        )
        config["/simulation/time/dt"] = "1 us"
        config["/simulation/time/max"] = "10 ms"

        print(yaml.dump(config.tree))
        raise typer.Exit(1)
