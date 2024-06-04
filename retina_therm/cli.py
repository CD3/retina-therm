import multiprocessing
import pprint
import sys
from pathlib import Path
from typing import Annotated, List

import numpy
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
from retina_therm import greens_functions, multi_pulse_builder, units, utils

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


def get_rendered(key: str, config: fspathtree, replace_spaces_char: str | None = None):
    val = config[key].format(c=config, id=config_utils.get_id(config))
    if replace_spaces_char is not None:
        val = val.replace(" ", str(replace_spaces_char))

    return val


def get_output_streams(config: fspathtree):
    stdout = sys.stdout
    stderr = sys.stderr
    config["cmd"] = invoked_subcommand
    if config.get("/simulation/stdout", None) is not None:
        stdout_path = Path(
            get_rendered("/simulation/stdout", config, replace_spaces_char="_")
        )
        if stdout_path.parent != Path():
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(stdout_path, "w")
    if config.get("/simulation/stderr", None) is not None:
        stderr_path = Path(
            get_rendered("/simulation/stderr", config, replace_spaces_char="_")
        )
        if stderr_path.parent != Path():
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr = open(stderr_path, "w")

    return stdout, stderr


def render_and_setup_output_file(output_filename, ctx):
    """Generate an output filename using the given context and create any parent directories if needed."""
    # render filename using context
    output_filename = output_filename.format(**ctx).replace(" ", "_")
    output_path = Path(output_filename)
    # if output file is in a subdirectory, create the directory
    if output_path.parent != Path():
        output_path.parent.mkdir(exist_ok=True, parents=True)

    return output_path


def load_config(config_file: Path, overrides: list[str]):
    global config_filename_stem
    config_filename_stem = config_file.stem
    if not config_file.exists():
        raise typer.Exit(f"File '{config_file}' not found.")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config = fspathtree(config)

    for item in overrides:
        k, v = [tok.strip() for tok in item.split("=", maxsplit=1)]
        if k not in config:
            sys.stderr.write(
                f"Warning: {k} was not in the config file, so it is being set, not overriden."
            )
        config[k] = v

    batch_leaves = config_utils.get_batch_leaves(config)
    config["batch/leaves"] = batch_leaves
    configs = config_utils.batch_expand(config)
    for c in configs:
        for k in c.get_all_leaf_node_paths():
            if type(c[k]) is str and c[k].lower() in ["none", "null"]:
                c[k] = None
        config_utils.compute_missing_parameters(c)

    return configs


def relaxation_time_job(config: fspathtree) -> None:
    stdout, stderr, datout = get_output_streams(config)

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

    stdout.write(f"Looking for {threshold} thermal relaxation time.\n")
    stdout.write(f"Peak temperature is {mp.nstr(Tp, 5)}\n")
    stdout.write(f"Looking for time to {mp.nstr(Tth, 5)}\n")
    i = 1
    while T > threshold * Tp:
        i *= 2
        t = i * dt
        T = G(z, t)
    i_max = i
    i_min = i / 2
    stdout.write(f"Relaxation time bracketed: [{i_min*dt},{i_max*dt}]\n")

    t = utils.bisect(lambda t: G(z, r, t) - Tth, i_min * dt, i_max * dt)
    t = sum(t) / 2
    T = G(z, r, t)

    stdout.write(f"time: {mp.nstr(mp.mpf(t), 5)}\n")
    stdout.write(f"Temperature: {mp.nstr(T, 5)}\n")


@app.command()
def relaxation_time(
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


def impulse_response_job(config):
    stdout, stderr, datout = get_output_streams(config)

    G = greens_functions.MultiLayerGreensFunction(config.tree)
    threshold = config["impulse_response/threshold"]
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
    T = G(z, r, t)
    Tp = T
    Tth = threshold * Tp

    i = 1
    while (T > threshold * Tp) and (t < tmax):
        datout.write(f"{t} {T}\n")
        i += 1
        t = i * dt
        T = G(z, r, t)
    datout.write(f"{t} {T}\n\n")


@app.command()
def impulse_response(
    config_file: Path,
    dps: Annotated[
        int,
        typer.Option(help="The precision to use for calculations when mpmath is used."),
    ] = 100,
    threshold: Annotated[float, typer.Option()] = 0.01,
    override: Annotated[
        list[str],
        typer.Option(
            help="key=val string to override a configuration parameter. i.e. --parameter 'simulation/time/dt=2 us'"
        ),
    ] = [],
):
    mp.dps = dps

    configs = load_config(config_file, override)
    # we are going to run each configuration in parallel
    jobs = []
    # create the jobs to run
    for config in configs:
        config["impulse_response/threshold"] = threshold
        jobs.append(
            multiprocessing.Process(target=impulse_response_job, args=(config,))
        )
    # run the jobs
    for job in jobs:
        job.start()
    # wait on the jobs
    for job in jobs:
        job.join()


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
        config_id = config_utils.get_id(config)
        config = compute_tissue_properties(config)

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

        if config.get("simulation/time/ts", None) is None:
            dt = config.get("simulation/time/dt", "1 us")
            dt = units.Q_(dt).to("s").magnitude
            tmin = config.get("simulation/time/min", "0 second")
            tmin = units.Q_(tmin).to("s").magnitude
            tmax = config.get("simulation/time/max", "10 second")
            tmax = units.Q_(tmax).to("s").magnitude
            t = numpy.arange(tmin, tmax, dt)
        else:
            t = numpy.array(
                [
                    units.Q_(time).to("s").magnitude
                    for time in config["simulation/time/ts"]
                ]
            )

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
            "this_file_stem": Path(config["this_file"]).stem,
            "c": config,
        }

        output_paths = {}
        for k in ["simulation/output_file", "simulation/output_config_file"]:
            filename = config.get(k, None)
            output_paths[k + "_path"] = Path("/dev/stdout")
            if filename is not None:
                filename = filename.format(**ctx).replace(" ", "_")
                path = Path(filename)
                output_paths[k + "_path"] = path
                if path.parent != Path():
                    path.parent.mkdir(parents=True, exist_ok=True)

        output_paths["simulation/output_config_file_path"].write_text(
            yaml.dump(config.tree)
        )
        numpy.savetxt(output_paths["simulation/output_file_path"], numpy.c_[t, T])
        self.status.emit("done.")


@app.command()
def temperature_rise(
    config_files: Path,
    ids: Annotated[
        List[str],
        typer.Option(
            help="Only run simulation for configurations with ID in the given list."
        ),
    ] = None,
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

    configs = config_utils.load_configs(config_files)
    config_ids = list(map(config_utils.get_id, configs))
    if print_ids:
        for _id in config_ids:
            print(_id)
        raise typer.Exit(0)

    if len(ids) == 0:
        ids = config_ids

    configs_to_run = list(filter(lambda c: config_utils.get_id(c) in ids, configs))
    if len(configs_to_run) == 0:
        rich.print("[orange]No configurations matched list of IDs to run[/orange]")
    if len(configs_to_run) > 1:
        # disable printing status information when we are processing multiple configurations
        console.print = lambda *args, **kwargs: None

    controller = parallel_jobs.Controller(TemperatureRiseProcess)
    controller.run(configs_to_run)
    controller.stop()

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
        N = 100

        stdout = sys.stdout
        stdout = sys.stderr
        if config.get("simulation/stdout", None) is not None:
            stdout = open(
                config["simulation/stdout"].format(
                    cmd="multipulse-microcavitation-threshold", c=config
                ),
                "w",
            )
        if config.get("simulation/stderr", None) is not None:
            stdout = open(
                config["simulation/stderr"].format(
                    cmd="multipulse-microcavitation-threshold", c=config
                ),
                "w",
            )

        config["laser/E0"] = "1 W/cm^2"  # override user power
        G = greens_functions.MultiLayerGreensFunction(config.tree)
        z = config.get("sensor/z", "0 um")
        z = units.Q_(z).to("cm").magnitude
        r = config.get("sensor/r", "0 um")
        r = units.Q_(r).to("cm").magnitude

        T = numpy.zeros([N])

        for i in range(1, len(T)):
            T[i] = T[i - 1] + G(z, r, t0 * i)

        for n in range(1, N):
            H = (m * T0 - m * Tnuc) / (1 - m * units.Q_(T[n - 1], "K/(J/cm^2)"))
            stdout.write(f"{n} {H}\n")


class MultiplePulseProcess(parallel_jobs.JobProcess):
    def run_job(self, config):
        config_id = config_utils.get_id(config)
        self.status.emit(
            "Loading base temperature history for building multiple-pulse history."
        )
        input_file = Path(config["input_file"])
        data = numpy.loadtxt(config["input_file"])
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
        if "/T" not in config:
            config["/T"] = "{:~}".format(units.Q_(t[-1], "s"))

        if "/t0" in config:
            t0 = units.Q_(config["/t0"])
            prf = (1 / t0).to("Hz")
            T = units.Q_(config["/T"])
            config["/prf"] = f"{prf:~}"

            if "/N" not in config:
                N = (T * prf).magnitude
                config["/N"] = f"{N:.2f}"

            N = units.Q_(config["/N"])

            arrival_time = units.Q_(0, "s")
            n = 0
            while arrival_time.to("s").magnitude < t[-1] and n < N:
                contributions.append(
                    {"arrival_time": arrival_time.to("s").magnitude, "scale": 1}
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
            "input_filename_stem": input_file.stem,
            "tau": config.get("/tau", "None").replace(" ", "_"),
            "t0": config.get("/t0", "None").replace(" ", "_"),
            "prf": config.get("/prf", "None").replace(" ", "_"),
            "PRF": config.get("/prf", "None").replace(" ", "_"),
            "N": config.get("/N", "None"),
            "n": config.get("/N", "None"),
            "T": config.get("/T", "None").replace(" ", "_"),
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
        numpy.savetxt(output_paths["output_file_path"], data)


@app.command()
def multiple_pulse(
    config_files: List[Path],
    ids: Annotated[
        List[str],
        typer.Option(
            help="Only run simulation for configurations with ID in the given list."
        ),
    ] = None,
    print_ids: Annotated[
        bool,
        typer.Option(
            help="Load configuration file(s) and print a list of the config IDs."
        ),
    ] = False,
):
    configs = config_utils.load_configs(config_files)
    config_ids = list(map(config_utils.get_id, configs))
    if print_ids:
        for _id in config_ids:
            print(_id)
        raise typer.Exit(0)

    if len(ids) == 0:
        ids = config_ids

    configs_to_run = list(filter(lambda c: config_utils.get_id(c) in ids, configs))
    if len(configs_to_run) > 1:
        # disable printing status information when we are processing multiple configurations
        console.print = lambda *args, **kwargs: None

    controller = parallel_jobs.Controller(MultiplePulseProcess)
    controller.run(configs_to_run)
    controller.stop()


def truncate_temperature_history_file_job(config):
    file = config["file"]
    threshold = config["threshold"]

    print(f"Truncating temperature_history in {file}.")
    data = numpy.loadtxt(file)
    threshold = units.Q_(threshold)
    if threshold.check(""):
        Tmax = max(data[:, 1])
        Tthreshold = threshold * Tmax
    elif threshold.check("K"):
        Tthreshold = threshold

    if data[-1, 1] > Tthreshold:
        print(f"  {file} already trucated...skipping.")
        return

    idx = numpy.argmax(numpy.flip(data[:, 1]) > Tthreshold)
    print(f"  Saving trucated history back to {file}.")
    numpy.savetxt(file, data[:-idx, :])


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
    with multiprocessing.Pool() as pool:
        for r in pool.imap_unordered(truncate_temperature_history_file_job, configs):
            pass


@app.command()
def print_config_ids(
    config_files: List[Path],
):
    """Print IDs of configuration in CONFIG_FILES. Useful for determining if a configuration has already been ran."""
    configs = config_utils.load_configs(config_files)
    config_ids = list(map(config_utils.get_id, configs))
    for _id in config_ids:
        print(_id)


@app.command()
def expand_configs(
    config_files: List[Path],
    output: Annotated[
        str, typer.Option(help="Format string for generating output file names.")
    ] = "{this_file_stem}-{i}.{this_file_ext}",
):
    """Expand configuration files and write out to separate files."""
    configs = config_utils.load_configs(config_files)
    for i, config in enumerate(configs):
        this_file = Path(config["this_file"])
        ctx = {
            "this_file": str(this_file),
            "this_file_stem": this_file.stem,
            "this_file_suffix": this_file.suffix,
            "this_file_ext": this_file.suffix[1:],
            "config_id": config_utils.get_id(config),
            "i": i,
        }

        output_file = Path(output.format(**ctx))
        if output_file in config_files:
            raise RuntimeError(
                f"Output config file {output_file} would overwrite an input file."
            )

        if output_file.parent != Path():
            output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(yaml.dump(config.tree))
