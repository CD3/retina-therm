"""
Schemas for parsing and validating model configurations.
"""

import math
import pathlib
from typing import Annotated, Any, List, Literal, TypeVar, Union

import numpy
from pydantic import (AfterValidator, AliasChoices, BaseModel, BeforeValidator,
                      Field, GetCoreSchemaHandler, PlainSerializer,
                      WithJsonSchema, model_validator)
from pydantic_core import CoreSchema, core_schema

from .units import Q_


def make_quantity(qstr):
    """
    Make a quantity from a string.

    For _most_ quantity strings we can just do Q_(qstr). However,
    this does not work for offset units.
    """
    v, u = qstr.split(maxsplit=1)
    v = float(v)
    u = u.strip()
    # handle inverse units that are given as the
    # value divided by a unit. i.e. "10 / s"
    # is "10 1/s"
    if len(u) > 0 and u[0] == "/":
        u = "1" + u
    return Q_(v, u)


QuantityWithUnit = lambda U, names=None, desc=None: Annotated[
    str,
    AfterValidator(lambda x: make_quantity(x).to(U)),
    PlainSerializer(lambda x: f"{x:~}" if x is not None else "null", return_type=str),
    WithJsonSchema({"type": "string"}, mode="serialization"),
    Field(
        alias=(
            AliasChoices(*names if type(names) is list else names)
            if names is not None
            else None
        ),
        description=desc,
    ),
]


class LayerConfig(BaseModel):
    thickness: QuantityWithUnit("cm", names=["thickness", "d"])
    position: QuantityWithUnit("cm", names=["position", "z0"])
    absorption_coeffcient: QuantityWithUnit(
        "1/cm", names=["absorption_coeffcient", "mua"]
    )


class LaserConfig(BaseModel):
    profile: Annotated[
        Literal["gaussian"] | Literal["flattop"] | Literal["1d"],
        BeforeValidator(lambda x: x.lower().replace(" ", "")),
    ] = "flattop"
    one_over_e_radius: Union[QuantityWithUnit("cm") | None] = Field(default=None)
    irradiance: QuantityWithUnit("W/cm^2", names=["irradiance", "E0"])

    # create a model validator that will check that one_over_e_radius is
    # given if profile is not 1D
    @model_validator(mode="after")
    def check_R(self) -> "LaserConfig":
        if self.profile != "1d" and self.one_over_e_radius is None:
            raise ValueError(
                f"'one_over_e_radius' must be given for '{self.profile}' profile."
            )
        return self


class CWLaserConfig(LaserConfig):
    start: QuantityWithUnit("s") = Field(default="0 s", validate_default=True)
    duration: QuantityWithUnit("s") = Field(default="1 year", validate_default=True)


class PulsedLaserConfig(CWLaserConfig):
    pulse_duration: QuantityWithUnit("s")
    pulse_period: QuantityWithUnit("s") = Field(default="1 year", validate_default=True)


class ThermalPropertiesConfig(BaseModel):
    rho: QuantityWithUnit("g/cm^3")
    c: QuantityWithUnit("J/g/K")
    k: QuantityWithUnit("W/cm/K")


class LargeBeamAbsorbingLayerGreensFunctionConfig(LayerConfig):
    rho: QuantityWithUnit("g/cm^3")
    c: QuantityWithUnit("J/g/K")
    k: QuantityWithUnit("W/cm/K")
    irradiance: QuantityWithUnit("W/cm^2", names=["irradiance", "E0"])

    with_units: bool = False
    use_multi_precision: bool = False
    use_approximations: bool = True


class FlatTopBeamAbsorbingLayerGreensFunctionConfig(
    LargeBeamAbsorbingLayerGreensFunctionConfig
):
    one_over_e_radius: QuantityWithUnit("cm")


class GaussianBeamAbsorbingLayerGreensFunctionConfig(
    FlatTopBeamAbsorbingLayerGreensFunctionConfig
):
    pass


class PrecisionConfig(BaseModel):
    use_multi_precision: bool = False
    use_approximations: bool = True
    with_units: bool = False


class MultiLayerGreensFunctionConfig(BaseModel):
    laser: LaserConfig
    thermal: ThermalPropertiesConfig
    layers: List[LayerConfig]

    class SimulationConfig(PrecisionConfig):
        pass

    simulation: SimulationConfig


class CWRetinaLaserExposureConfig(MultiLayerGreensFunctionConfig):
    laser: CWLaserConfig


class PulsedRetinaLaserExposureConfig(MultiLayerGreensFunctionConfig):
    laser: PulsedLaserConfig


class MultiplePulseContribution(BaseModel):
    arrival_time: QuantityWithUnit("s")
    scale: float


class MultiplePulseCmdConfig(BaseModel):
    input_file: pathlib.Path
    output_file: pathlib.Path
    output_config_file: pathlib.Path

    tau: QuantityWithUnit("s") = None
    t0: QuantityWithUnit("s") = None
    N: int = None

    contributions: List[MultiplePulseContribution] = []

    # create a model validator that will check t0 and N were given
    # if 'contributions' field was _not_ given
    @model_validator(mode="after")
    def check_pulse_config(self) -> "MultiplePulseCmdConfig":
        if len(self.contributions) == 0:
            if self.t0 is None and self.N is None:
                raise ValueError(
                    f"Regular pulse configuration parameters, 't0' and 'N', must be given if 'contributions' is not given."
                )
        return self
