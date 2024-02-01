"""
Schemas for parsing and validating model configurations.
"""


from typing import Annotated, Any, TypeVar

from pydantic import (AfterValidator, BaseModel, Field, GetCoreSchemaHandler,
                      PlainSerializer, WithJsonSchema)
from pydantic_core import CoreSchema, core_schema

from .units import Q_

QuantityWithUnit = lambda U: Annotated[
    str,
    AfterValidator(lambda x: Q_(x).to(U)),
    PlainSerializer(lambda x: f"{x:~}", return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]

Quantity = Annotated[
    str,
    AfterValidator(lambda x: Q_(x)),
    PlainSerializer(lambda x: f"{x:~}", return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
]


class Layer(BaseModel):
    d: QuantityWithUnit("cm")
    z0: QuantityWithUnit("cm")
    mua: QuantityWithUnit("1/cm")


class LargeBeamAbsorbingLayerGreensFunctionConfig(BaseModel):
    mua: QuantityWithUnit("1/cm")
    rho: QuantityWithUnit("g/cm^3")
    c: QuantityWithUnit("J/g/K")
    k: QuantityWithUnit("W/cm/K")
    d: QuantityWithUnit("cm")
    z0: QuantityWithUnit("cm")
    E0: QuantityWithUnit("W/cm^2")
