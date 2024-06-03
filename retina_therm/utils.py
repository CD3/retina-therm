import copy
import importlib.resources
import itertools
import math

import numpy
import scipy
from fspathtree import fspathtree

marcum_q_wasm_module_file = importlib.resources.path(
    "retina_therm.wasm", "marcum_q.wasm"
)
try:
    import wasmer

    have_wasmer_module = True
except:
    have_wasmer_module = False

have_marcum_q_wasm_module = False
if have_wasmer_module:
    if marcum_q_wasm_module_file.exists():
        wasm_store = wasmer.Store()
        wasm_module = wasmer.Module(wasm_store, marcum_q_wasm_module_file.read_bytes())

        wasi_version = wasmer.wasi.get_version(wasm_module, strict=True)
        wasi_env = wasmer.wasi.StateBuilder("marcum_q").finalize()
        wasm_import_object = wasi_env.generate_import_object(wasm_store, wasi_version)

        wasm_instance = wasmer.Instance(wasm_module, wasm_import_object)
        have_marcum_q_wasm_module = True
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
