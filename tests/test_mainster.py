import pytest

from retina_therm import greens_functions, units, utils


def test_compare_to_mainter_temperature_rises_calculations_for_flattop_profile():
    # data from mainster 1970 paper, table I
    data = {
        "10 um": [
            (1e-10, 3e-8),
            (1e-9, 3e-7),
            (1e-8, 3e-6),
            (1e-7, 3e-5),
            (1e-6, 2.9e-4),
            (1e-5, 2.6e-3),
            (1e-4, 1.6e-2),
            (1e-3, 4.1e-2),
            (1e-2, 5.5e-2),
            (1e-1, 6.1e-2),
            (1e0, 6.3e-2),
            (1e1, 6.4e-2),
            # (1e2, 6.4e-2),
            # (1e3, 6.4e-2),
        ],
        "25 um": [
            (1e-10, 3e-8),
            (1e-9, 3e-7),
            (1e-8, 3e-6),
            (1e-7, 3e-5),
            (1e-6, 2.9e-4),
            (1e-5, 2.6e-3),
            (1e-4, 1.8e-2),
            (1e-3, 9.2e-2),
            (1e-2, 2.0e-1),
            (1e-1, 2.6e-1),
            (1e0, 2.8e-1),
            (1e1, 2.9e-1),
            # (1e2, 2.9e-1),
            # (1e3, 2.9e-1),
        ],
        "50 um": [
            (1e-10, 3e-8),
            (1e-9, 3e-7),
            (1e-8, 3e-6),
            (1e-7, 3e-5),
            (1e-6, 2.9e-4),
            (1e-5, 2.6e-3),
            (1e-4, 1.8e-2),
            (1e-3, 9.7e-2),
            (1e-2, 2.8e-1),
            (1e-1, 4.1e-1),
            (1e0, 4.5e-1),
            (1e1, 4.7e-1),
            # (1e2, 4.7e-1),
            # (1e3, 4.7e-1),
        ],
        "100 um": [
            (1e-10, 3e-8),
            (1e-9, 3e-7),
            (1e-8, 3e-6),
            (1e-7, 3e-5),
            (1e-6, 2.9e-4),
            (1e-5, 2.6e-3),
            (1e-4, 1.8e-2),
            (1e-3, 9.9e-2),
            (1e-2, 4.6e-1),
            (1e-1, 1.2),
            (1e0, 1.6),
            (1e1, 1.7),
            # (1e2, 1.8),
            # (1e3, 1.8),
        ],
        "500 um": [
            (1e-10, 3e-8),
            (1e-9, 3e-7),
            (1e-8, 3e-6),
            (1e-7, 3e-5),
            (1e-6, 2.9e-4),
            (1e-5, 2.6e-3),
            (1e-4, 1.8e-2),
            (1e-3, 9.9e-2),
            (1e-2, 4.7e-1),
            (1e-1, 5.4),
            (1e0, 7.2),
            (1e1, 7.7),
            # (1e2, 7.7),
            # (1e3, 7.7),
        ],
        "500 um": [
            (1e-10, 3e-8),
            (1e-9, 3e-7),
            (1e-8, 3e-6),
            (1e-7, 3e-5),
            (1e-6, 2.9e-4),
            (1e-5, 2.6e-3),
            (1e-4, 1.8e-2),
            (1e-3, 9.9e-2),
            (1e-2, 4.7e-1),
            (1e-1, 2.1),
            (1e0, 5.4),
            (1e1, 7.2),
            # (1e2, 7.7),
            # (1e3, 7.7),
        ],
        "1000 um": [
            (1e-10, 3e-8),
            (1e-9, 3e-7),
            (1e-8, 3e-6),
            (1e-7, 3e-5),
            (1e-6, 2.9e-4),
            (1e-5, 2.6e-3),
            (1e-4, 1.8e-2),
            (1e-3, 9.9e-2),
            (1e-2, 4.7e-1),
            (1e-1, 2.1),
            (1e0, 7.2),
            (1e1, 1.3e1),
            # (1e2, 1.6e1),
            # (1e3, 1.6e1),
        ],
    }

    # not sure why the error increases with spot size here
    allowed_error = {
        "10 um": 1e-2,
        "25 um": 2e-2,
        "50 um": 8e-2,
        "100 um": 22e-2,
        "500 um": 22e-2,
        "1000 um": 36e-2,
    }

    E0 = units.Q_("1 W/cm^2")

    for size in ["10 um", "25 um", "50 um", "100 um", "500 um", "1000 um"]:
        config = {
            "simulation": {
                "use_approximations": True,
                "use_multi_precision": False,
            },
            "laser": {
                "pulse_duration": "1000 s",
                "E0": str(E0),
                "profile": "flattop",
                "one_over_e_radius": str(units.Q_(size)),
            },
            "thermal": {
                "rho": "1 g/cm^3",
                "c": "1 cal / K / g",
                "k": "1.5e-3 cal / K / s / cm",
            },
            "layers": [
                {"mua": "310 1/cm", "d": "10 um", "z0": "0 um"},
                {"mua": "53 1/cm", "d": "100 um", "z0": "10 um"},
            ],
        }
        G = greens_functions.PulsedRetinaLaserExposure(config)

        t = [item[0] for item in data[size]]

        T = G.temperature_rise(
            z=units.Q_("1 um").to("cm").magnitude,
            r=units.Q_("0 cm").to("cm").magnitude,
            t=t,
            method="quad",
        )
        # compute v/E0 (temperature rise normalized to irradiance) to compare to the values reported by mainster
        v_over_E0 = list(
            map(lambda T: (units.Q_(T, "K") / E0).to("K/ (cal/cm^2/s)").magnitude, T)
        )
        # for t_,T_ in zip(t,v_over_E0):
        #     print(f"{t_:e} {T_:e}")
        # print()
        # error = rms deviations
        error = (
            sum(
                map(
                    lambda e: (e[0] - e[1]) ** 2,
                    zip([item[1] for item in data[size]], v_over_E0),
                )
            )
            / len(v_over_E0)
        ) ** 0.5
        assert error < allowed_error[size]
