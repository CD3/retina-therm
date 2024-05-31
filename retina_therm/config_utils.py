import copy
import hashlib
import itertools
import json
import pathlib
import typing

import numpy
import pydantic.v1.utils
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


def load_configs(config_files: typing.List[pathlib.Path]):
    """
    Load a set of configuration files (YAML files) and return a set of configuration objects (fspathtree instances).
    A configuration file may contain multiple document, in which case the file is assumed to
    represent a set of configurations where the first document contains common parameters to all
    configurations, and the other documents contain overrides.

    After all files have been loaded, @batch parameters are expanded, which can result in more configuration
    objects being generated.
    """
    if type(config_files) not in [list]:
        config_files = [config_files]
    configs = list()

    for file in config_files:
        full_text = pathlib.Path(file).read_text()
        doc_texts = full_text.split("\n---\n")
        if any(map(lambda t: t == "", map(lambda t: t.strip(), doc_texts))):
            raise RuntimeError(f"Configuration '{file}' contained an empty document.")

        if len(doc_texts) < 1:
            raise RuntimeError(f"No yaml document found in '{file}'.")

        if len(doc_texts) == 1:
            # if this is the only document in the config file,
            # then we just want to add it to the set of configs.
            config = yaml.safe_load(doc_texts[0])
            config = fspathtree(config)
            config["this_file"] = str(file)
            configs.append(config)
        else:
            # if there are more then on document, then we want
            # to treat the first as a "base" configuration and
            # generate a configuration sequence for eac of the
            # following documents
            base_config = yaml.safe_load(doc_texts[0])
            for text in doc_texts[1:]:
                config = copy.deepcopy(base_config)
                c = yaml.safe_load(text)
                # we can't use python's builtin update here
                # because it will replace entire branch of config tree
                config = pydantic.v1.utils.deep_update(config, c)
                # config.update(c)
                config = fspathtree(config)
                config["this_file"] = str(file)
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


def get_id(config: fspathtree, strip_keys=["/this_file"]):
    """Return a unique id for the given configuration object."""
    # make a copy of the config with only keys not in the strip list
    c = fspathtree()

    def filt(path):
        if str(path) in strip_keys:
            return False
        return True

    # we use a filter on the get_all_leaf_node_paths(...) here for more flexability
    for p in config.get_all_leaf_node_paths(predicate=filt):
        c[p] = config[p]

    text = json.dumps(c.tree, sort_keys=True).replace(" ", "")
    return hashlib.md5(text.encode("utf-8")).hexdigest()
