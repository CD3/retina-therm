import multiprocessing
import pprint
import sys
from pathlib import Path
from typing import Annotated

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
from retina_therm import greens_functions, units, utils

from . import config_utils, utils

app = typer.Typer()


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
    datout = sys.stdout
    config["cmd"] = invoked_subcommand
    config["config_file/stem"] = config_filename_stem
    if config.get("simulation/datout", None) is not None:
        datout = open(
            get_rendered("simulation/datout", config, replace_spaces_char="_"),
            "w",
        )
    if config.get("simulation/stdout", None) is not None:
        stdout = open(
            get_rendered("simulation/stdout", config, replace_spaces_char="_"),
            "w",
        )
    if config.get("simulation/stderr", None) is not None:
        stderr = open(
            get_rendered("simulation/stderr", config, replace_spaces_char="_"),
            "w",
        )

    return stdout, stderr, datout


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


def temperature_rise_job(config):
    config = compute_tissue_properties(config)
    stdout, stderr, datout = get_output_streams(config)
    stdout.write(str(config.tree) + "\n")
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
        tmax = config.get("simulation/time/max", "10 second")
        tmax = units.Q_(tmax).to("s").magnitude
        t = numpy.arange(0, tmax, dt)
    else:
        t = numpy.array(
            [units.Q_(time).to("s").magnitude for time in config["simulation/time/ts"]]
        )

    method = config.get("simulation/temperature_rise/method", "quad")

    print("Computing temperature rise")
    T = G.temperature_rise(z, r, t, method=method)
    print("Writing temperature rise")
    for i in range(len(T)):
        datout.write(f"{t[i]} {T[i]}\n")


@app.command()
def temperature_rise(
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
    method: Annotated[
        str, typer.Option(help="Integration method to use.")
    ] = "undefined",
    list_methods: Annotated[
        bool, typer.Option(help="List the avaiable integration methods.")
    ] = False,
):
    if list_methods:
        print("Available inegration methods:")
        for m in temperature_rise_integration_methods:
            print("  ", m)
        raise typer.Exit(0)
    if method not in temperature_rise_integration_methods + ["undefined"]:
        rich.print(f"[red]Unrecognized integration method '{method}'[/red]")
        rich.print(
            f"[red]Please use one of {', '.join(temperature_rise_integration_methods)}[/red]"
        )
        raise typer.Exit(1)

    mp.dps = dps

    configs = load_config(config_file, override)
    jobs = []
    for config in configs:
        if method != "undefined":
            config["simulation/temperature_rise/method"] = method
        jobs.append(
            multiprocessing.Process(target=temperature_rise_job, args=(config,))
        )
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()


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
