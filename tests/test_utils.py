import pprint
import pytest
import pathlib
import numpy

import retina_therm.config_utils
import retina_therm.units
import retina_therm.utils

from .unit_test_utils import working_directory

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
    assert retina_therm.units.Q_(config["laser/R"]).to("cm").magnitude == pytest.approx(
        0.0005
    )

    ##########################

    config = retina_therm.utils.fspathtree({"laser": {"Phi": "1 mW", "D": "10 um"}})
    retina_therm.config_utils.compute_missing_parameters(config)

    assert retina_therm.units.Q_(config["laser/E0"]).magnitude == pytest.approx(
        1 / (3.14159 * 5**2)
    )
    assert retina_therm.units.Q_(config["laser/E0"]).to(
        "W/cm**2"
    ).magnitude == pytest.approx(1e5 / (3.14159 * 5**2))

    ##########################

    config = retina_therm.utils.fspathtree(
        {"laser": {"H": "1 mJ/cm^2", "D": "10 um", "duration": "2 s"}}
    )
    retina_therm.config_utils.compute_missing_parameters(config)

    assert retina_therm.units.Q_(config["laser/E0"]).magnitude == pytest.approx(0.5)
    assert retina_therm.units.Q_(config["laser/E0"]).to(
        "W/cm**2"
    ).magnitude == pytest.approx(0.5e-3)

    ##########################

    config = retina_therm.utils.fspathtree(
        {"laser": {"Q": "1 mJ", "D": "10 um", "duration": "2 s"}}
    )
    retina_therm.config_utils.compute_missing_parameters(config)

    assert retina_therm.units.Q_(config["laser/E0"]).magnitude == pytest.approx(
        0.5 / (3.14159 * 5**2)
    )
    assert retina_therm.units.Q_(config["laser/E0"]).to(
        "W/cm**2"
    ).magnitude == pytest.approx(1e5 * 0.5 / (3.14159 * 5**2))


def test_marcum_q_function():
    # computed using WolframAlpha https://wolframalpha.com
    evaluations = [
        ((1, 0, 0), 1),
        ((2, 0, 0), 1),
        ((2, 1, 0), 1),
        ((2, 1, 0), 1),
        (
            (1, 0, 1),
            0.6065306597126334236037995349911804534419181354871869556828921587,
        ),
        (
            (1, 2, 1),
            0.9181076963694060039105695602622025530636609822389841572133252640,
        ),
        (
            (1, 1, 1),
            0.7328798037968202182509507647816049993664329559143995840198057465,
        ),
        (
            (1, 1, 2),
            0.2690120600359099966785169592202710874213375007448733841550744652,
        ),
    ]

    for args, value in evaluations:
        assert retina_therm.utils.MarcumQFunction(*args) == pytest.approx(value)

#
# def test_marcum_q_function_performance():
#     import matplotlib.pyplot as plt
#     N = 10
#     duration = timeit.Timer(lambda : retina_therm.utils.MarcumQFunction(1,0,1)).timeit(number=N)
#     marcum_runtime=duration/N
#     print(">>>",marcum_runtime)
#     duration = timeit.Timer(lambda : numpy.exp(-1)).timeit(number=N)
#     exp_runtime=duration/N
#     print(">>>",exp_runtime)
#     print("marcum/exp:",marcum_runtime/exp_runtime)
#
#     pass

    # x = numpy.arange(0,5,0.01)
    # f1 = numpy.array([1-retina_therm.utils.MarcumQFunction(1,1,2**0.5*b) for b in x])
    # f2 = numpy.array([1-retina_therm.utils.MarcumQFunction(1,2,2**0.5*b) for b in x])
    # f3 = numpy.array([ 1-numpy.exp(-b**2) for b in x])
    #
    # plt.plot(x,f1,label="f1")
    # plt.plot(x,f2,label="f2")
    # plt.plot(x,f3,label="f3")
    # plt.legend(loc="upper right")
    # plt.show()


def test_writing_arrays_to_file(tmp_path):
    with working_directory(tmp_path):

        x = numpy.array([1,2,3])
        y = numpy.array([3,4,5])

        assert not pathlib.Path('data.txt').exists()
        retina_therm.utils.write_to_file( "data.txt", numpy.c_[x,y], fmt="txt" )
        assert pathlib.Path('data.txt').exists()

        data = retina_therm.utils.read_from_file( "data.txt", fmt="txt" )

        assert data[0,0] == 1
        assert data[1,0] == 2
        assert data[2,0] == 3

        assert data[0,1] == 3
        assert data[1,1] == 4
        assert data[2,1] == 5


        assert not pathlib.Path('data.hdf5').exists()
        retina_therm.utils.write_to_file( "data.hdf5", numpy.c_[x,y], fmt="hdf5" )
        assert pathlib.Path('data.hdf5').exists()

        data = retina_therm.utils.read_from_file( "data.hdf5", fmt="hdf5" )

        assert data[0,0] == 1
        assert data[1,0] == 2
        assert data[2,0] == 3

        assert data[0,1] == 3
        assert data[1,1] == 4
        assert data[2,1] == 5



