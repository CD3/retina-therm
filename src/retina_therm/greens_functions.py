import copy
import math

import numpy
import scipy
from mpmath import mp
from tqdm import tqdm

from .config import *
from .signals import Signal
from .units import *
from .utils import MarcumQFunction


class LargeBeamAbsorbingLayerGreensFunction:
    def __init__(
        self, config: dict | LargeBeamAbsorbingLayerGreensFunctionConfig
    ) -> None:
        if type(config) == dict:
            config = LargeBeamAbsorbingLayerGreensFunctionConfig(**config)

        self.mua = config.absorption_coeffcient
        self.k = config.k
        self.rho = config.rho
        self.c = config.c
        self.E0 = config.irradiance
        self.d = config.thickness
        self.z0 = config.position
        self.alpha = self.k / self.rho / self.c

        self.with_units = config.with_units
        self.use_multi_precision = config.use_multi_precision
        self.use_approximations = config.use_approximations
        if not self.with_units:
            for param in ["mua", "k", "rho", "c", "E0", "d", "z0", "alpha"]:
                setattr(self, param, getattr(self, param).magnitude)

        if self.use_multi_precision:
            for param in ["mua", "k", "rho", "c", "E0", "d", "z0", "alpha"]:
                setattr(self, param, mp.mpf(getattr(self, param)))
            self.erf = mp.erf
            self.sqrt = mp.sqrt
            self.exp = mp.exp
        elif self.with_units:
            self.erf = scipy.special.erf
            self.sqrt = numpy.sqrt
            self.exp = numpy.exp
        else:
            # the math module implementations seem to be faster based on test_benchmarks...
            # hyperfine benchmarks of the cli also seem to indicate a speedup
            self.erf = math.erf
            self.sqrt = math.sqrt
            self.exp = math.exp

    def __call__(
        self, z: float | mp.mpf | Q_, r: float | mp.mpf | Q_, tp: float | mp.mpf | Q_
    ) -> float | mp.mpf | Q_:
        if self.use_approximations and False:
            # Large time approximation...
            # This approximates the exponential in the integrand (before integration)
            # and then integrates. The problem is that mua is not accounted for at all,
            # so it is possible for this approximatin to be "valid", but not good if mua is large.
            # We we are disabling it
            if tp > 0:
                arg1 = (z - self.z0) ** 2 / (4 * self.alpha * tp)
                arg2 = (z - (self.z0 + self.d)) ** 2 / (4 * self.alpha * tp)
                thresh = Q_(0.1, "")
                # use the approximation for exp(-(1/x)**2) if 1/x
                if not self.with_units:
                    thresh = thresh.magnitude
                if arg1 < thresh and arg2 < thresh:
                    # some terms that are used multiple times
                    exp_mua_d = self.exp(-self.mua * self.d)
                    four_alpha_tp = 4 * self.alpha * tp

                    # see writeup, there three factors that need to be multiplied
                    # together, were the last factor is a sum of four terms.

                    const_factor = self.E0 / self.rho / self.c
                    time_factor = 1 / self.sqrt(numpy.pi * four_alpha_tp)
                    term1 = 1 - exp_mua_d

                    factor21 = 2 * z**2 / four_alpha_tp
                    factor22 = term1
                    term2 = factor21 * factor22

                    factor31 = 2 * z / four_alpha_tp
                    factor32_term1 = self.z0 + 1 / self.mua
                    factor32_term2 = (self.z0 + self.d + 1 / self.mua) * exp_mua_d
                    term3 = factor31 * (factor32_term1 - factor32_term2)

                    factor41 = 1 / four_alpha_tp
                    factor42_term1 = (
                        self.z0**2 + 2 * self.z0 / self.mua + 2 / self.mua**2
                    )
                    factor42_term2 = (
                        (self.z0 - self.d) ** 2
                        + 2 * (self.z0 + self.d) / self.mua
                        + 2 / self.mua**2
                    ) * exp_mua_d
                    term4 = factor41 * (factor42_term1 - factor42_term2)

                    low_order_approx = const_factor * time_factor * term1
                    high_order_correction = (
                        const_factor * time_factor * (-term2 + term3 - term4)
                    )
                    approx = low_order_approx + high_order_correction
                    if approx < 0:
                        return low_order_approx

                    return approx

        term1 = self.mua * self.E0 / self.rho / self.c / 2
        term2 = self.exp(-self.mua * (z - self.z0))
        if tp == 0:
            return term1 * term2

        if self.use_approximations:
            # Asymptotic approximation for erf
            # This approximation is valid when the arguments to the error function
            # are greater than about 2. But we don't really need to use it that early.
            # we get reasonable error when the arguments are >= 4.
            A = self.mua * (self.alpha * tp) ** 0.5
            B = (self.z0 - z) / (4 * self.alpha * tp) ** 0.5
            C = self.d / (4 * self.alpha * tp) ** 0.5
            # We will get an overflow if A^2 > about 700
            if A + B + C > 4 and A + B > 4:
                B2 = B * B
                C2 = C * C
                _2AB = 2 * A * B
                _2AC = 2 * A * C
                _2BC = 2 * B * C

                factor1 = (
                    self.exp(-B2 - _2AB) / (A + B) / self.sqrt(numpy.pi)
                )  # * ( 1 - 1/2/(A+B)**2)
                factor2 = (
                    self.exp(-B2 - C2 - _2AB - _2AC - _2BC)
                    / (A + B + C)
                    / self.sqrt(numpy.pi)
                )  # * ( 1 - 1/2/(A+B+C)**2)
                term3 = factor1 - factor2

                return term1 * term2 * term3

        # this has a (good) chance of overflowing if we are using floats
        # if it is going to overflow then we want to set it to inf. this is what numpy does
        try:
            term3 = self.exp(self.alpha * tp * self.mua**2)
        except OverflowError:
            term3 = float("inf")
        arg1 = (self.z0 + self.d - z) / self.sqrt(4 * self.alpha * tp) + self.sqrt(
            self.alpha * tp
        ) * self.mua
        arg2 = (self.z0 - z) / self.sqrt(4 * self.alpha * tp) + self.sqrt(
            self.alpha * tp
        ) * self.mua
        if self.with_units:
            arg1 = arg1.magnitude
            arg2 = arg2.magnitude
        term4 = self.erf(arg1) - self.erf(arg2)

        return term1 * term2 * term3 * term4


class FlatTopBeamAbsorbingLayerGreensFunction(LargeBeamAbsorbingLayerGreensFunction):
    """
    The Green's function for a flat top beam is just the same as a large beam with an additional
    term for the radial part.
    """

    def __init__(
        self, config: dict | FlatTopBeamAbsorbingLayerGreensFunctionConfig
    ) -> None:
        if type(config) == dict:
            config = FlatTopBeamAbsorbingLayerGreensFunctionConfig(**config)
        super().__init__(config.model_dump())

        self.R = config.one_over_e_radius

        if not self.with_units:
            for param in ["R"]:
                setattr(self, param, getattr(self, param).magnitude)

        if self.use_multi_precision:
            for param in ["R"]:
                setattr(self, param, mp.mpf(getattr(self, param)))

    def __call__(
        self, z: float | mp.mpf, r: float | mp.mpf, tp: float | mp.mpf = None
    ) -> float | mp.mpf:
        # special conditions
        # if the sensor is outside of the beam at t = 0,
        # the temperature rise will be zero
        if tp == 0 and r > self.R:
            return 0.0
        zfactor = super().__call__(z, r, tp)

        # if the sensor is inside the beam, at t = 0,
        # the temperature rise will be the same as on axis
        if tp == 0:
            return zfactor

        if r == 0:
            # If we want the temperature on the z axis, it is _much_
            # faster to call the exp(...) function instead of MarcumQFunction.
            rfactor = 1 - self.exp(-(self.R**2) / 4 / self.alpha / tp)
        else:
            # If we are calculating the temperature off axis, we have no choice
            # but to call the expensive function
            # TODO: add support for calling a WASM-commpiled version of this function. Initial testing indicates
            #       it could be 10x faster.
            rfactor = 1 - MarcumQFunction(
                1,
                r / self.sqrt(2 * self.alpha * tp),
                self.R / self.sqrt(2 * self.alpha * tp),
            )

        return zfactor * rfactor


class GaussianBeamAbsorbingLayerGreensFunction(FlatTopBeamAbsorbingLayerGreensFunction):
    def __init__(
        self, config: dict | GaussianBeamAbsorbingLayerGreensFunctionConfig
    ) -> None:
        if type(config) == dict:
            config = GaussianBeamAbsorbingLayerGreensFunctionConfig(**config)
        super().__init__(config.model_dump())

    def __call__(
        self, z: float | mp.mpf, r: float | mp.mpf, tp: float | mp.mpf = None
    ) -> float | mp.mpf:
        zfactor = super().__call__(z, r, tp)

        if r == 0:
            rfactor = 1 / (1 + 4 * self.alpha * tp / self.R**2)
        else:
            tmp1 = 1 / (1 + 4 * self.alpha * tp / self.R**2)
            tmp2 = self.R**2 / 4 / self.alpha / tp

            rfactor = tmp1 * self.exp(tmp2 * (tmp1 - 1))

        return zfactor * rfactor


class MultiLayerGreensFunction:
    def __init__(self, config: dict | MultiLayerGreensFunctionConfig) -> None:
        if type(config) == dict:
            config = MultiLayerGreensFunctionConfig(**config)

        self.with_units = config.simulation.with_units
        self.use_multi_precision = config.simulation.use_multi_precision
        self.use_approximations = config.simulation.use_approximations

        if self.use_multi_precision:
            self.erf = mp.erf
            self.sqrt = mp.sqrt
            self.exp = mp.exp
        elif self.with_units:
            self.erf = scipy.special.erf
            self.sqrt = numpy.sqrt
            self.exp = numpy.exp
        else:
            # the math module implementations seem to be faster based on test_benchmarks...
            # hyperfine benchmarks of the cli also seem to indicate a speedup
            self.erf = math.erf
            self.sqrt = math.sqrt
            self.exp = math.exp

        self.layers = []

        E0 = config.laser.irradiance
        for layer in sorted(config.layers, key=lambda l: l.position.magnitude):
            # create a config dict that we will pass to the layer
            # and fill in the keys needed by a layer
            # thank you pydantic...
            layer_config = layer.model_dump()
            layer_config.update(config.thermal.model_dump())
            layer_config.update(config.laser.model_dump())
            layer_config.update(config.simulation.model_dump())
            # override irradiance
            layer_config.update(
                {
                    # make sure we use "irradiance" here.
                    # "E0" is an alias, but if both are given "irradiance" will be used,
                    # and "irradiance" is already in the dict.
                    "irradiance": str(E0),
                }
            )

            if config.laser.profile == "1d":
                G = LargeBeamAbsorbingLayerGreensFunction(layer_config)
            if config.laser.profile == "flattop":
                G = FlatTopBeamAbsorbingLayerGreensFunction(layer_config)
            if config.laser.profile == "gaussian":
                G = GaussianBeamAbsorbingLayerGreensFunction(layer_config)

            self.layers.append(G)

            # Need to reduce the incident irradiance according to Beer's Law
            mua = layer.absorption_coeffcient
            d = layer.thickness

            if not self.with_units:
                mua = mua.magnitude
                d = d.magnitude

            if self.use_multi_precision:
                mua = mp.mpf(mua)
                d = mp.mpf(d)

            E0 = E0 * self.exp(-mua * d)

        for i in range(len(self.layers)):
            if i > 0:
                if self.layers[i].z0 < self.layers[i - 1].z0 + self.layers[i - 1].d:
                    raise RuntimeError(
                        f"ERROR: Layer {i} overlaps with layer {i-1}. z_{i} = {self.layers[i].z0}, z_{i-1} + d_{i-1} = {self.layers[i-1].z0+ self.layers[i-1].d}"
                    )

    def __call__(
        self, z: float | mp.mpf, r: float | mp.mpf, tp: float | mp.mpf = None
    ) -> float | mp.mpf:
        return sum([G(z, r, tp) for G in self.layers])


class GreensFunctionIntegrator:
    def __init__(self, G) -> None:
        self.G = G
        self.progress = Signal()
        self.status = Signal()


class GreensFunctionTrapezoidIntegrator(GreensFunctionIntegrator):
    def __init__(self, G) -> None:
        super().__init__(G)
        self.dt = Q_(0.1, "us")

    def temperature_rise_on_grid(self, z, r, tmin, tmax, dt):
        """Compute the temperature rise at uniformly spaced times so that we can build the temperature rise caused by an exposure."""
        t = numpy.arange(tmin, tmax + 2 * dt, dt)
        T = numpy.vectorize(lambda x: self.G(z, r, x))(t)
        T = scipy.integrate.cumulative_trapezoid(T, t)
        return t[:-1], T

    def temperature_rise(
        self,
        z: float | mp.mpf,
        r: float | mp.mpf,
        ts: list[float | mp.mpf],
        config: dict,
    ):
        """Compute the temperature rise caused by an exposure descibed in config."""
        tmin = min(ts)
        tmax = max(ts)
        dt = self.dt.to("s").magnitude
        t, dTheta = self.temperature_rise_on_grid(z, r, tmin, tmax, dt)

        ton = Q_(config.get("ton", "0 s")).to("s").magnitude
        tau = Q_(config.get("tau", "1 year")).to("s").magnitude
        t0 = Q_(config.get("t0", "1 year")).to("s").magnitude
        T = Q_(config.get("T", "1 year")).to("s").magnitude

        dTheta_sp = numpy.zeros([len(t)])
        i_start = int(ton / dt)
        i_stop = min(int((ton + tau) / dt), len(t) - 1)

        for i in range(i_start, i_stop + 1):
            dTheta_sp[i] = dTheta[i - i_start]
        for i in range(i_stop + 1, len(t)):
            # dT = int_{t-tau}^t G(t') dt'
            dTheta_sp[i] = (
                dTheta[i - i_start]
                - dTheta[i - i_stop]  # i - i_start - (i_stop-i_start)
            )
        dTheta_mp = dTheta_sp
        N = math.ceil(T / t0)
        for n in range(1, N):
            i_start = int((ton + n * t0) / dt)
            dTheta_mp[i_start:] += dTheta_sp[:-i_start]

        _is = numpy.searchsorted(t, ts)
        Theta = dTheta_mp[_is]

        return Theta


class GreensFunctionQuadIntegrator(GreensFunctionIntegrator):
    def __init__(self, G) -> None:
        super().__init__(G)
        self.max_subinterval_range = Q_(0.1, "s")

    def temperature_rise(
        self,
        z: float | mp.mpf,
        r: float | mp.mpf,
        ts: list[float | mp.mpf],
        config: dict,
    ):
        ton = Q_(config.get("ton", "0 s")).to("s").magnitude
        tau = Q_(config.get("tau", "1 year")).to("s").magnitude
        t0 = Q_(config.get("t0", "1 year")).to("s").magnitude
        T = Q_(config.get("T", "1 year")).to("s").magnitude

        tmin = ton
        tmax = max(ts)

        max_subinterval_range = self.max_subinterval_range.to("s").magnitude
        num_subintervals = 1
        subinterval_range = (tmax - tmin) / num_subintervals
        while subinterval_range > max_subinterval_range:
            num_subintervals += 1
            subinterval_range = (tmax - tmin) / num_subintervals

        def f(tp):
            return self.G(z, r, tp)

        subinterval_values = numpy.zeros([num_subintervals])
        for i in range(num_subintervals):
            a = i * subinterval_range
            b = a + subinterval_range
            subinterval_values[i] = scipy.integrate.quad(f, a, b)[0]

        def calc_integral(a, b):
            if a < 0:
                a = 0
            if b < 0:
                b = 0
            il = int((a - tmin) / subinterval_range)
            iu = int((b - tmin) / subinterval_range)
            val = sum(subinterval_values[il + 1 : iu])
            if iu == il:
                val += scipy.integrate.quad(f, a, b)[0]
            else:
                val += scipy.integrate.quad(f, a, tmin + (il + 1) * subinterval_range)[
                    0
                ]
                val += scipy.integrate.quad(f, tmin + iu * subinterval_range, b)[0]
            return val

        dTheta = numpy.zeros([len(ts)])
        N = math.ceil((T - ton) / t0)
        num_iters = len(ts)
        for i, t in enumerate(ts):
            n = math.ceil((t - ton) / t0)
            for j in range(min(n, N)):
                if t - j * t0 < tau:
                    dTheta[i] += calc_integral(0, (t - ton) - j * t0)
                else:
                    dTheta[i] += calc_integral(
                        (t - ton) - j * t0 - tau, (t - ton) - j * t0
                    )
            self.progress.emit(i + 1, num_iters)

        return dTheta


class CWRetinaLaserExposure:
    """
    Class for configuring and computing the temperature rise from a CW exposure to a retina model.
    """

    def __init__(self, config: dict) -> None:
        if type(config) == dict:
            config = CWRetinaLaserExposureConfig(**config)
        self.G = MultiLayerGreensFunction(config.model_dump())

        self.start = config.laser.start
        self.duration = config.laser.duration

        self.progress = Signal()

    def make_integrator_config(self):
        config = {
            "tau": self.duration,
            "ton": self.start,
        }
        return config

    def temperature_rise(
        self,
        z: float | mp.mpf,
        r: float | mp.mpf,
        t: list[float] | list[mp.mpf],
        method="trap",
    ):
        Integrator = None
        if method == "trap":
            Integrator = GreensFunctionTrapezoidIntegrator(self.G)
        if method == "quad":
            Integrator = GreensFunctionQuadIntegrator(self.G)
        Integrator.progress.connect(lambda i, n: self.progress.emit(i, n))

        return Integrator.temperature_rise(z, r, t, self.make_integrator_config())


class PulsedRetinaLaserExposure(CWRetinaLaserExposure):
    def __init__(self, config: dict) -> None:
        if type(config) == dict:
            config = PulsedRetinaLaserExposureConfig(**config)
        super().__init__(config.model_dump())

        self.exposure_duration = self.duration
        self.pulse_duration = config.laser.pulse_duration
        self.pulse_period = config.laser.pulse_period

    def make_integrator_config(self):
        config = {
            "tau": self.pulse_duration,
            "t0": self.pulse_period,
            "T": self.exposure_duration,
            "ton": self.start,
        }
        return config
