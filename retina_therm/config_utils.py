import copy
import hashlib
import itertools
import pathlib
import typing

import numpy
import yaml
from fspathtree import fspathtree

from . import units, utils


def get_batch_leaves(config: fspathtree):
    """
    Return a list of keys in a fpathtree (nested dict/list) that are marked
    as batch.
    """
    batch_leaves = {}
    for leaf in config.get_all_leaf_node_paths():
        if leaf.parent.parts[-1] == "@batch":
            batch_leaves[str(leaf.parent.parent)] = (
                batch_leaves.get(leaf.parent.parent, 0) + 1
            )
    return list(batch_leaves.keys())


def batch_expand(config: fspathtree):
    configs = []

    batch_leaves = get_batch_leaves(config)

    for vals in itertools.product(*[config[leaf + "/@batch"] for leaf in batch_leaves]):
        instance = copy.deepcopy(config)
        for i, leaf in enumerate(batch_leaves):
            instance[leaf] = vals[i]
        configs.append(instance)

    return configs


def load_configs(config_files: typing.List[pathlib.Path], include_base=False):
    if type(config_files) not in [list]:
        config_files = [config_files]
    configs = list()

    base_config = yaml.safe_load(pathlib.Path(config_files[0]).read_text())
    if include_base:
        config = copy.deepcopy(base_config)
        config = fspathtree(config)
        configs.append(config)
    for file in config_files[1:]:
        config = copy.deepcopy(base_config)
        c = yaml.safe_load(pathlib.Path(file).read_text())
        config.update(c)
        config = fspathtree(config)
        configs.append(config)

    # expand batch parameters
    configs = list(
        itertools.chain(*map(lambda c: batch_expand(c), configs)),
    )

    # Do we want to allow overrides?
    # If so, where should they be overriden? Here? Or before batch expansion?
    # for item in overrides:
    #     k, v = [tok.strip() for tok in item.split("=", maxsplit=1)]
    #     if k not in config:
    #         sys.stderr.write(
    #             f"Warning: {k} was not in the config file, so it is being set, not overriden."
    #         )
    #     config[k] = v
    return configs


def compute_missing_parameters(config):
    """
    Compute values for missing parameters in the config. For example, if laser irradiance
    is not given, but a laser power and beam diameter is, then we can compute the irradiance.
    """
    if "laser/R" not in config:
        if "laser/D" in config:
            config["laser/R"] = str(units.Q_(config["laser/D"]) / 2)

    if "laser/E0" not in config:
        if "laser/Q" in config:
            if "laser/pulse_duration" in config or "laser/duration" in config:
                t = units.Q_(
                    config.get("laser/pulse_duration", config["laser/duration"])
                )
                Q = units.Q_(config["laser/Q"])
                Phi = Q / t
                config["laser/Phi"] = str(Phi)
        if "laser/Phi" in config and "laser/R" in config:
            Phi = units.Q_(config["laser/Phi"])
            R = units.Q_(config["laser/R"])
            E0 = Phi / (numpy.pi * R**2)
            config["laser/E0"] = str(E0)
        if "laser/H" in config:
            if "laser/pulse_duration" in config or "laser/duration" in config:
                t = units.Q_(
                    config.get("laser/pulse_duration", config["laser/duration"])
                )
                H = units.Q_(config["laser/H"])
                E0 = H / t
                config["laser/E0"] = str(E0)

    missing_params = [param for param in ["laser/R", "laser/E0"] if param not in config]
    if len(missing_params) > 0:
        raise RuntimeError(
            f"Could not find or compute required parameters: {', '.join(missing_params)}"
        )


def get_id(config: fspathtree):
    text = str(config).replace(" ", "")
    return hashlib.md5(text.encode("utf-8")).hexdigest()
