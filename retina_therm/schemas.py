"""
Schemas for parsing and validating model configurations.
"""

import math
import pathlib
from typing import Annotated, Any, List, Literal, TypeVar, Union

import numpy
from pydantic import (AfterValidator, BaseModel, BeforeValidator, Field,
                      GetCoreSchemaHandler, PlainSerializer, WithJsonSchema,
                      model_validator)
from pydantic_core import CoreSchema, core_schema

from .units import Q_

QuantityWithUnit = lambda U: Annotated[
    str,
    AfterValidator(lambda x: Q_(x).to(U)),
    PlainSerializer(lambda x: f"{x:~}" if x is not None else "null", return_type=str),
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
    ] = "flattop"
    R: Union[QuantityWithUnit("cm"), None] = Field(default=None)
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
        if self.profile != "1d":
            if self.R:
                self.D_ = self.R * 2
            else:
                self.R = self.D_ / 2

        return self


class CWLaser(Laser):
    start: QuantityWithUnit("s") = Field(default="0 s", validate_default=True)
    duration: QuantityWithUnit("s") = Field(default="1 year", validate_default=True)


class PulsedLaser(CWLaser):
    pulse_duration: QuantityWithUnit("s")
    pulse_period: QuantityWithUnit("s") = Field(default="1 year", validate_default=True)


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
    use_multi_precision: bool = False
    use_approximations: bool = True
    with_units: bool = False


class MultiLayerGreensFunctionConfig(BaseModel):
    laser: Laser
    thermal: ThermalProperties
    layers: List[Layer]

    class Simulation(PrecisionConfig):
        pass

    simulation: Simulation


class CWRetinaLaserExposureConfig(MultiLayerGreensFunctionConfig):
    laser: CWLaser


class PulsedRetinaLaserExposureConfig(MultiLayerGreensFunctionConfig):
    laser: PulsedLaser


# class MultiplePulseContribution:
#     arrival_time: QuantityWithUnit("s")
#     scale: QuantityWithUnit("dimensionless")


# class MultiplePulseCmdConfig(BaseModel):
#     input_file: pathlib.Path
#     output_file: pathlib.Path
#     tau: QuantityWithUnit("s")
#     T: QuantityWithUnit("s")
#     t0: QuantityWithUnit("s")
#     contributions: List[MultiplePulseContribution] = []
