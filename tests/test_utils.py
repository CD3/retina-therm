import pprint
import pytest
import pathlib
import numpy

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



