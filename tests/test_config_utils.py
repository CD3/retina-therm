import os
import pathlib

import numpy
import pytest

from retina_therm import config_utils


def test_batch_expansion_single_vars():
    config = config_utils.fspathtree(
        {"one": 1, "two": {"@batch": [2, 3]}, "three": {"vals": {"@batch": [3, 4]}}}
    )

    leaves = config_utils.get_batch_leaves(config)

    assert len(leaves) == 2
    assert leaves[0] == "/two"
    assert leaves[1] == "/three/vals"

    configs = config_utils.batch_expand(config)

    assert len(configs) == 4
    assert configs[0]["/one"] == 1
    assert configs[0]["/two"] == 2
    assert configs[0]["/three/vals"] == 3

    assert configs[1]["/one"] == 1
    assert configs[1]["/two"] == 2
    assert configs[1]["/three/vals"] == 4

    assert configs[2]["/one"] == 1
    assert configs[2]["/two"] == 3
    assert configs[2]["/three/vals"] == 3

    assert configs[3]["/one"] == 1
    assert configs[3]["/two"] == 3
    assert configs[3]["/three/vals"] == 4


def test_loading_single_config(tmp_path):
    orig_path = pathlib.Path().absolute()
    os.chdir(tmp_path)
    config_text = """
one: 1
two: 2
    """

    pathlib.Path("config.yml").write_text(config_text)

    configs = config_utils.load_configs("config.yml",include_base=True)
    assert len(configs) == 1

    assert configs[0]["/one"] == 1
    assert configs[0]["/two"] == 2
    os.chdir(orig_path)


def test_loading_two_config(tmp_path):
    orig_path = pathlib.Path().absolute()
    os.chdir(tmp_path)
    config_text = """
one: 1
two: 2
    """
    pathlib.Path("config.yml").write_text(config_text)

    config_text = """
two: 4
three: 3
    """
    pathlib.Path("config2.yml").write_text(config_text)

    configs = config_utils.load_configs(["config.yml", "config2.yml"],include_base=True)

    assert len(configs) == 2

    assert configs[0]["/one"] == 1
    assert configs[0]["/two"] == 2
    assert "/three" not in configs[0]

    assert configs[1]["/one"] == 1
    assert configs[1]["/two"] == 4
    assert configs[1]["/three"] == 3

    os.chdir(orig_path)


def test_loading_config_with_batch_parameter(tmp_path):
    orig_path = pathlib.Path().absolute()
    os.chdir(tmp_path)
    config_text = """
one: 1
two: 
    '@batch':
      - 2
      - 4
    """
    pathlib.Path("config.yml").write_text(config_text)

    configs = config_utils.load_configs("config.yml",include_base=True)

    assert len(configs) == 2

    assert configs[0]["/one"] == 1
    assert configs[0]["/two"] == 2

    assert configs[1]["/one"] == 1
    assert configs[1]["/two"] == 4

    os.chdir(orig_path)
