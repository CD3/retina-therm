"""
Schemas for parsing and validating model configurations.
"""


from typing import Annotated, Any, List, Literal, TypeVar

from pydantic import (AfterValidator, BaseModel, BeforeValidator, Field,
                      GetCoreSchemaHandler, PlainSerializer, WithJsonSchema, model_validator)
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


class Laser(BaseModel):
    profile: Annotated[
        Literal["gaussian"] | Literal["flattop"],
        BeforeValidator(lambda x: x.lower().replace(" ", "")),
    ]
    R: QuantityWithUnit("cm") = None

    # allow user to specify diameter D instead of radius R
    # create a field named "D_"" that will be used internally.
    # user will pass "D"
    D_: QuantityWithUnit("cm") = Field(alias="D",default=None)

    # create property named "D" that the user can use if they want.
    @property
    def D(self):
        return self.R*2
    @D.setter
    def D(self,val):
        self.R = val/2

    # create a model validator that will check that one of "R" or "D"
    # were passed in and set the other.
    @model_validator(mode='after')
    def check_R_or_D(self) -> 'Laser':
        print(self.R,self.D_)
        if self.R is None and self.D_ is None:
                raise ValueError("One of 'R' or 'D' must be given.")
        if self.R:
            self.D_ = self.R * 2
        else:
            self.R = self.D_ / 2

        return self





class ThermalProperties(BaseModel):
    rho: QuantityWithUnit("g/cm^3")
    c: QuantityWithUnit("J/g/K")
    k: QuantityWithUnit("W/cm/K")


class LargeBeamAbsorbingLayerGreensFunctionConfig(BaseModel):
    mua: QuantityWithUnit("1/cm")
    rho: QuantityWithUnit("g/cm^3")
    c: QuantityWithUnit("J/g/K")
    k: QuantityWithUnit("W/cm/K")
    d: QuantityWithUnit("cm")
    z0: QuantityWithUnit("cm")
    E0: QuantityWithUnit("W/cm^2")


class FlatTopBeamAbsorbingLayerGreensFunctionConfig(
    LargeBeamAbsorbingLayerGreensFunctionConfig
):
    R: QuantityWithUnit("cm")


class GaussianBeamAbsorbingLayerGreensFunctionConfig(
    FlatTopBeamAbsorbingLayerGreensFunctionConfig
):
    pass


class MultiLayerGreensFunctionConfig(BaseModel):
    laser: Laser
    thermal: ThermalProperties
    layers: List[Layer]
