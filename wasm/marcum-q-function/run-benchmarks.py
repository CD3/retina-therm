import pathlib
import timeit

import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

import numpy
import scipy
import wasmer

wasm_store = wasmer.Store()
wasm_module = wasmer.Module(wasm_store, pathlib.Path("./marcum_q.wasm").read_bytes())

wasi_version = wasmer.wasi.get_version(wasm_module, strict=True)
wasi_env = wasmer.wasi.StateBuilder("marcum_q").finalize()
wasm_import_object = wasi_env.generate_import_object(wasm_store, wasi_version)

wasm_instance = wasmer.Instance(wasm_module, wasm_import_object)


def wasm_MarcumQFunction(nu, a, b):
    return wasm_instance.exports.MarcumQFunction(nu, a, b)


def wasm_MarcumQFunction10(b):
    return wasm_instance.exports.MarcumQFunction10(b)


def python_MarcumQFunction(nu, a, b):
    return 1 - scipy.stats.ncx2.cdf(b**2, 2 * nu, a**2)


def python_MarcumQFunction10(b):
    return numpy.exp(-b * b / 2)


assert (
    abs(wasm_MarcumQFunction(1.0, 0.0, 1.0) - python_MarcumQFunction(1, 0, 1)) < 0.0001
)
assert abs(python_MarcumQFunction(1, 0, 1) - python_MarcumQFunction10(1)) < 0.0001
assert abs(python_MarcumQFunction10(1.0) - wasm_MarcumQFunction10(1.0)) < 0.0001

N = 1000
duration_wasm = timeit.Timer(lambda: wasm_MarcumQFunction(1.0, 0.0, 1.0)).timeit(
    number=N
)
duration_wasm = Q_(duration_wasm / N, "s")
duration_wasm_2 = timeit.Timer(lambda: wasm_MarcumQFunction10(1.0)).timeit(number=N)
duration_wasm_2 = Q_(duration_wasm_2 / N, "s")
duration_python = timeit.Timer(lambda: python_MarcumQFunction(1.0, 0.0, 1.0)).timeit(
    number=N
)
duration_python = Q_(duration_python / N, "s")
duration_python_2 = timeit.Timer(lambda: python_MarcumQFunction10(1.0)).timeit(number=N)
duration_python_2 = Q_(duration_python_2 / N, "s")

print("WASM M-Q:", duration_wasm.to("us"))
print("WASM M-Q 10:", duration_wasm_2.to("us"))
print("Python M-Q:", duration_python.to("us"))
print("Python M-Q 10:", duration_python_2.to("us"))
