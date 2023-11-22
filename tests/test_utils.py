import pprint

import pytest

import retina_therm.config_utils
import retina_therm.utils
import retina_therm.units


def test_bisect():
    f = lambda x: 2 * x + 1

    with pytest.raises(RuntimeError):
        retina_therm.utils.bisect(f, 0, 1)

    with pytest.raises(RuntimeError):
        retina_therm.utils.bisect(f, -10, -9)

    assert retina_therm.utils.bisect(f, -10, 10)[0] < -0.5
    assert retina_therm.utils.bisect(f, -10, 10)[1] > -0.5
    assert sum(retina_therm.utils.bisect(f, -10, 10)) / 2 == pytest.approx(-0.5)


def test_batch_leave_detection():
    config = retina_therm.utils.fspathtree(
        {
            "b": 20,
            "a": {"@batch": [1, 2, 3]},
            "l1": {"b": {"@batch": ["one", "two"]}, "c": 10},
        }
    )
    leaves = [str(l) for l in retina_therm.config_utils.get_batch_leaves(config)]

    assert len(leaves) == 2
    assert "/a" in leaves
    assert "/l1/b" in leaves


def test_expand_batch_single_batch_var():
    config = retina_therm.utils.fspathtree({"a": {"@batch": [1, 2, 3]}})
    configs = retina_therm.config_utils.batch_expand(config)

    assert len(configs) == 3
    assert configs[0]["a"] == 1
    assert configs[1]["a"] == 2
    assert configs[2]["a"] == 3


def test_expand_batch_two_batch_var():
    config = retina_therm.utils.fspathtree(
        {"a": {"@batch": [1, 2, 3]}, "b": {"@batch": [4, 5]}}
    )
    configs = retina_therm.config_utils.batch_expand(config)

    assert len(configs) == 6
    assert configs[0]["a"] == 1
    assert configs[0]["b"] == 4
    assert configs[1]["a"] == 1
    assert configs[1]["b"] == 5
    assert configs[2]["a"] == 2
    assert configs[2]["b"] == 4
    assert configs[3]["a"] == 2
    assert configs[3]["b"] == 5


def test_expand_batch_with_quantities():
    config = retina_therm.utils.fspathtree({"a": {"@batch": ["1 us", "2 us"]}})
    configs = retina_therm.config_utils.batch_expand(config)

    assert len(configs) == 2
    assert configs[0]["a"] == "1 us"
    assert configs[1]["a"] == "2 us"


def test_compute_missing_parameters():
    config = retina_therm.utils.fspathtree(

        {"laser": {"E0": "1 mW/cm**2", "R": "10 um"}}
    )
    retina_therm.config_utils.compute_missing_parameters(config)

    config["laser/E0"] == "1 mW/cm**2"
    config["laser/R"] == "10 um"



    ##########################

    config = retina_therm.utils.fspathtree(

        {"laser": {"E0": "1 mW/cm**2", "D": "10 um"}}
    )
    retina_therm.config_utils.compute_missing_parameters(config)

    config["laser/E0"] == "1 mW/cm**2"
    assert retina_therm.units.Q_(config["laser/R"]).magnitude == pytest.approx(5)
    assert retina_therm.units.Q_(config["laser/R"]).to("cm").magnitude == pytest.approx(0.0005)



    ##########################

    config = retina_therm.utils.fspathtree(

        {"laser": {"Phi": "1 mW", "D": "10 um"}}
    )
    retina_therm.config_utils.compute_missing_parameters(config)

    assert retina_therm.units.Q_(config["laser/E0"]).magnitude == pytest.approx(1 / (3.14159*5**2))
    assert retina_therm.units.Q_(config["laser/E0"]).to("W/cm**2").magnitude == pytest.approx(1e5 / (3.14159*5**2))

    ##########################

    config = retina_therm.utils.fspathtree(

            {"laser": {"H": "1 mJ/cm^2", "D": "10 um", 'duration':'2 s'}}
    )
    retina_therm.config_utils.compute_missing_parameters(config)

    assert retina_therm.units.Q_(config["laser/E0"]).magnitude == pytest.approx(0.5)
    assert retina_therm.units.Q_(config["laser/E0"]).to("W/cm**2").magnitude == pytest.approx(0.5e-3)



    ##########################

    config = retina_therm.utils.fspathtree(

            {"laser": {"Q": "1 mJ", "D": "10 um", 'duration':'2 s'}}
    )
    retina_therm.config_utils.compute_missing_parameters(config)

    assert retina_therm.units.Q_(config["laser/E0"]).magnitude == pytest.approx(0.5 / (3.14159*5**2))
    assert retina_therm.units.Q_(config["laser/E0"]).to("W/cm**2").magnitude == pytest.approx(1e5*0.5 / (3.14159*5**2))
