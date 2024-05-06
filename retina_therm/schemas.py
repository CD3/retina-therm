"""
Schemas for parsing and validating model configurations.
"""

import math
from typing import Annotated, Any, List, Literal, TypeVar

import numpy
from pydantic import (AfterValidator, BaseModel, BeforeValidator, Field,
                      GetCoreSchemaHandler, PlainSerializer, WithJsonSchema,
                      model_validator)
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
    z0: QuantityWithUnit("cm") = None
    mua: QuantityWithUnit("1/cm")


class Laser(BaseModel):
    profile: Annotated[
        Literal["gaussian"] | Literal["flattop"] | Literal["1d"],
        BeforeValidator(lambda x: x.lower().replace(" ", "")),
    ]
    R: QuantityWithUnit("cm") = None
    E0: QuantityWithUnit("W/cm^2")

    # allow user to specify diameter D instead of radius R
    # create a field named "D_"" that will be used internally.
    # user will pass "D"
    D_: QuantityWithUnit("cm") = Field(alias="D", default=None)

    # create property named "D" that the user can use if they want.
    @property
    def D(self):
        return self.R * 2

    @D.setter
    def D(self, val):
        self.R = val / 2

    # create a model validator that will check that one of "R" or "D"
    # were passed in and set the other.
    @model_validator(mode="after")
    def check_R_or_D(self) -> "Laser":
        if self.profile != "1d" and self.R is None and self.D_ is None:
            raise ValueError(
                "One of 'R' or 'D' must be given for '{self.profile}' profile."
            )
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

    with_units: bool = False
    use_multi_precision: bool = False
    use_approximations: bool = True


class FlatTopBeamAbsorbingLayerGreensFunctionConfig(
    LargeBeamAbsorbingLayerGreensFunctionConfig
):
    R: QuantityWithUnit("cm")


class GaussianBeamAbsorbingLayerGreensFunctionConfig(
    FlatTopBeamAbsorbingLayerGreensFunctionConfig
):
    pass


class PrecisionConfig(BaseModel):
    use_multi_precision: bool
    use_approximations: bool
    with_units: bool


class MultiLayerGreensFunctionConfig(BaseModel):
    laser: Laser
    thermal: ThermalProperties
    layers: List[Layer]
    # simulation: PrecisionConfig


def get_AbsorbingLayerGreensFunctionConfig_json(
    config: MultiLayerGreensFunctionConfig, layer_index: int
) -> dict:
    assert layer_index < len(config.layers)

    E0 = config.laser.E0
    for i in range(len(config.layers)):
        if i == layer_index:
            break
        E0 *= math.exp(-(config.layers[i].mua * config.layers[i].d).to(""))

    return {
        "rho": str(config.thermal.rho),
        "c": str(config.thermal.c),
        "k": str(config.thermal.k),
        "d": str(config.layers[layer_index].d),
        "z0": str(config.layers[layer_index].z0),
        "mua": str(config.layers[layer_index].mua),
        "E0": str(E0),
        "R": str(config.laser.R),
    }


def make_LargeBeamAbsorbingLayerGreensFunctionConfig_from_MultiLayerGreensFunctionConfig(
    config: MultiLayerGreensFunctionConfig, layer_index: int
) -> LargeBeamAbsorbingLayerGreensFunctionConfig:
    return LargeBeamAbsorbingLayerGreensFunctionConfig(
        **get_AbsorbingLayerGreensFunctionConfig_json(config, layer_index)
    )


def make_FlatTopBeamAbsorbingLayerGreensFunctionConfig_from_MultiLayerGreensFunctionConfig(
    config: MultiLayerGreensFunctionConfig, layer_index: int
) -> FlatTopBeamAbsorbingLayerGreensFunctionConfig:
    return FlatTopBeamAbsorbingLayerGreensFunctionConfig(
        **get_AbsorbingLayerGreensFunctionConfig_json(config, layer_index)
    )


def make_GaussianBeamAbsorbingLayerGreensFunctionConfig_from_MultiLayerGreensFunctionConfig(
    config: MultiLayerGreensFunctionConfig, layer_index: int
) -> GaussianBeamAbsorbingLayerGreensFunctionConfig:
    return GaussianBeamAbsorbingLayerGreensFunctionConfig(
        **get_AbsorbingLayerGreensFunctionConfig_json(config, layer_index)
    )


def make_AbsorbingLayerGreensFunctionConfig_from_MultiLayerGreensFunctionConfig(
    config: MultiLayerGreensFunctionConfig, layer_index: int
) -> (
    GaussianBeamAbsorbingLayerGreensFunctionConfig
    | FlatTopBeamAbsorbingLayerGreensFunctionConfig
    | LargeBeamAbsorbingLayerGreensFunctionConfig
):
    if config.laser.profile == "gaussian":
        return make_GaussianBeamAbsorbingLayerGreensFunctionConfig_from_MultiLayerGreensFunctionConfig(
            config, layer_index
        )
    if config.laser.profile == "flattop":
        return make_FlatTopBeamAbsorbingLayerGreensFunctionConfig_from_MultiLayerGreensFunctionConfig(
            config, layer_index
        )
    if config.laser.profile == "1d":
        return make_LargeBeamAbsorbingLayerGreensFunctionConfig_from_MultiLayerGreensFunctionConfig(
            config, layer_index
        )
