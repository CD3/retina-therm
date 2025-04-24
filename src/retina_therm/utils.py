import copy
import importlib.resources
import itertools
import math
import pathlib
import struct

import h5py
import numpy
import scipy
from fspathtree import fspathtree

# marcum_q_wasm_module_file = importlib.resources.path(
#     "retina_therm.wasm", "marcum_q.wasm"
# )
# try:
#     import wasmer

#     have_wasmer_module = True
# except:
#     have_wasmer_module = False

# have_marcum_q_wasm_module = False
# if have_wasmer_module:
#     if marcum_q_wasm_module_file.exists():
#         wasm_store = wasmer.Store()
#         wasm_module = wasmer.Module(wasm_store, marcum_q_wasm_module_file.read_bytes())

#         wasi_version = wasmer.wasi.get_version(wasm_module, strict=True)
#         wasi_env = wasmer.wasi.StateBuilder("marcum_q").finalize()
#         wasm_import_object = wasi_env.generate_import_object(wasm_store, wasi_version)

#         wasm_instance = wasmer.Instance(wasm_module, wasm_import_object)
#         have_marcum_q_wasm_module = True
# DISABLE FOR NOW
# getting WASI error when tyring to run large batch configs
have_marcum_q_wasm_module = False


def bisect(f, a, b, tol=1e-8, max_iter=1000):
    lower = f(a)
    upper = f(b)
    sign = (upper - lower) / abs(upper - lower)
    if lower * upper >= 0:
        raise RuntimeError(
            f"Bracket [{a},{b}] does not contain a zero. f(a) and f(b) should have different signs but they were {lower} and {upper}."
        )

    mid = (a + b) / 2
    f_mid = f(mid)
    num_iter = 0
    while num_iter < max_iter and abs(f_mid) > tol:
        num_iter += 1
        a = mid if sign * f_mid < 0 else a
        b = mid if sign * f_mid > 0 else b
        mid = (a + b) / 2
        f_mid = f(mid)

    return (a, b)


def MarcumQFunction_PYTHON(nu, a, b):
    return 1 - scipy.stats.ncx2.cdf(b**2, 2 * nu, a**2)


if have_marcum_q_wasm_module:

    def MarcumQFunction_WASM(nu, a, b):
        ret = wasm_instance.exports.MarcumQFunction(float(nu), float(a), float(b))
        if math.isnan(ret):
            # fall back to python implementation if we get a nan
            ret = MarcumQFunction_PYTHON(nu, a, b)
        return ret

    MarcumQFunction = MarcumQFunction_WASM
else:
    MarcumQFunction = MarcumQFunction_PYTHON


def write_to_file(filepath: pathlib.Path, array: numpy.array, fmt="hdf5"):

    if fmt in ["txt"]:
        numpy.savetxt(filepath, array)
        return

    if fmt in ["hdf5"]:
        f = h5py.File(filepath, "w")
        f.create_dataset("retina-therm", data=array)
        f.close()
        return

    raise RuntimeError(f"Unrecognized format '{fmt}'")


def read_from_file(filepath: pathlib.Path, fmt="hdf5"):
    if fmt in ["txt"]:
        return numpy.loadtxt(filepath)

    if fmt in ["hdf5"]:
        f = h5py.File(filepath, "r")
        data = f["retina-therm"][:]
        f.close()
        return data

    raise RuntimeError(f"Unrecognized format '{fmt}'")


def read_Tvst_from_file_txt(filepath: pathlib.Path):
    return numpy.loadtxt(filepath)


def read_Tvst_from_file_hdf5(filepath: pathlib.Path):
    f = h5py.File(filepath, "r")
    data = f["retina-therm"][:]
    f.close()
    return data


def read_Tvst_from_file_rt(filepath: pathlib.Path):

    with open(filepath, "rb") as f:
        f.seek(0, 2)
        fs = f.tell()
        f.seek(0, 0)
        if fs % 8 > 0:
            raise RuntimeError(
                f"Invalid or corrupt file. rt binary file should contain a multiple of 8 bytes. {filepath} contains {fs} bytes."
            )
        N = int(fs / 8 - 1)
        dt = struct.unpack("d", f.read(8))[0]
        Tvst = numpy.zeros((N, 2))
        Tvst[:, 0] = numpy.arange(0, dt * N, dt)
        Tvst[:, 1] = list(
            map(lambda item: item[0], struct.iter_unpack("d", f.read(8 * N)))
        )
        # print(Tvst)
    return Tvst


def read_Tvst_from_file(filepath: pathlib.Path, fmt):
    if fmt in ["txt"]:
        return read_Tvst_from_file_txt(filepath)

    if fmt in ["hdf5"]:
        return read_Tvst_from_file_hdf5(filepath)

    if fmt in ["rt"]:
        return read_Tvst_from_file_rt(filepath)

    raise RuntimeError(f"Unrecognized format '{fmt}'")


def write_Tvst_to_file_txt(data: numpy.array, filepath: pathlib.Path):
    numpy.savetxt(filepath, data)


def write_Tvst_to_file_hdf5(data: numpy.array, filepath: pathlib.Path):
    f = h5py.File(filepath, "w")
    f.create_dataset("retina-therm", data=data)
    f.close()


def write_Tvst_to_file_rt(data: numpy.array, filepath: pathlib.Path):
    # check that dat is uniform
    # the difference between consecutive times should be the same, to within 1 picosecond
    diffs = numpy.diff(data[:, 0])
    if sum((diffs - diffs[0]) > 1e-9) > 0:
        raise RuntimeError(
            "time-temperature history must be uniformly spaced to save to 'rt' binary file."
        )

    with open(filepath, "wb") as f:
        f.write(diffs[0])
        T = copy.copy(data[:, 1])
        f.write(T)


def write_Tvst_to_file(data: numpy.array, filepath: pathlib.Path, fmt):
    if fmt in ["txt"]:
        return write_Tvst_to_file_txt(data, filepath)

    if fmt in ["hdf5"]:
        return write_Tvst_to_file_hdf5(data, filepath)

    if fmt in ["rt"]:
        return write_Tvst_to_file_rt(data, filepath)

    raise RuntimeError(f"Unrecognized format '{fmt}'")
