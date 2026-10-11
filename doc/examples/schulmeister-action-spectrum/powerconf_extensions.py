import pathlib

import powerconf
import tissue_properties
from powerconf.units import Q_
from tissue_properties.optical.absorption_coefficient import schulmeister
from tissue_properties.optical.ocular_transmission import cie203


def fto(q, unit):
    return q.to(unit).magnitude


def ito(q, unit):
    return int(q.to(unit).magnitude)


_RPE = schulmeister.RPE()
_Choroid = schulmeister.Choroid()
_Henle = schulmeister.HenlesFiberLayer()
_Transmission = cie203.total_transmission.TotalTransmission()


def RPE(wavelength):
    # we can't mix units from two different systems, so we have to turn wavelength
    # into a string to pass it in.
    return _RPE(str(wavelength))


def Choroid(wavelength):
    return _Choroid(str(wavelength))


def Henle(wavelength):
    return _Henle(str(wavelength))


def T(wavelength):
    return Q_(str(_Transmission(str(wavelength))))


def get_peak_temp(filename):
    Tmax = None
    T0 = None
    file = pathlib.Path(filename)
    if file.exists():
        for line in file.read_text().split("\n"):
            try:
                t, T = map(float, line.split())
                if T0 is None:
                    T0 = T
                if Tmax is None or T > Tmax:
                    Tmax = T

            except:
                pass

    if T0 == 0:
        Tmax += 310.15
    return Tmax


def get_threshold_peak_temp(filename):
    return get_peak_temp(filename + ".threshold")


def get_threshold_scale(filename):
    damage_file = pathlib.Path("damage.txt")
    threshold = None
    if damage_file.exists():
        for line in damage_file.read_text().split("\n"):
            if filename in line:
                cols = line.split("|")
                threshold = cols[-1]
                break

    if threshold is None:
        Tth = get_threshold_peak_temp(filename)
        Tp = get_peak_temp(filename)
        if Tth is not None and Tp is not None:
            threshold = (Tth - 310.15) / (Tp - 310.15)

    return threshold
