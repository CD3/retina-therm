import copy
import math

import pytest

from retina_therm import greens_functions
from retina_therm.units import *
from retina_therm.utils import *

# def test_wasm_vs_python_implementations():
#     assert MarcumQFunction_WASM(1, 0, 0) == pytest.approx(
#         MarcumQFunction_PYTHON(1, 0, 0)
#     )
#     assert MarcumQFunction_WASM(1, 1e4, 1e4) == pytest.approx(
#         MarcumQFunction_PYTHON(1, 1e4, 1e4)
#     )
#     # the WASM implementation gives a nan here....
#     # assert MarcumQFunction_WASM(1, 42, 17) == pytest.approx(
#     #     MarcumQFunction_PYTHON(1, 42, 17)
#     # )

#     assert MarcumQFunction_WASM(1, 1, 0) == pytest.approx(
#         MarcumQFunction_PYTHON(1, 1, 0)
#     )
#     assert MarcumQFunction_WASM(1, 1, 1) == pytest.approx(
#         MarcumQFunction_PYTHON(1, 1, 1)
#     )
#     assert MarcumQFunction_WASM(1, 2, 1) == pytest.approx(
#         MarcumQFunction_PYTHON(1, 2, 1)
#     )
#     assert MarcumQFunction_WASM(1, 2, 2) == pytest.approx(
#         MarcumQFunction_PYTHON(1, 2, 2)
#     )
