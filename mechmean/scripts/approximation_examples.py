#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approximations of effective stiffness of two-phase materials
"""

import numpy as np
import mechmean
import mechkit
from mechmean import utils
import pprint

if __name__ == "__main__":
    np.set_printoptions(linewidth=140, precision=3)

    inp = {
        "E_f": 73.0,
        "E_m": 3.4,
        "N4": np.array(
            [
                [4.09e-01, 1.48e-01, 1.03e-02, -2.20e-03, -1.86e-02, 3.52e-02],
                [1.48e-01, 2.51e-01, 6.50e-03, -2.00e-03, -5.50e-03, 3.11e-02],
                [1.03e-02, 6.50e-03, 9.70e-03, 8.00e-04, -1.20e-03, 4.00e-04],
                [-2.20e-03, -2.00e-03, 8.00e-04, 1.30e-02, 5.00e-04, -7.70e-03],
                [-1.86e-02, -5.50e-03, -1.20e-03, 5.00e-04, 2.06e-02, -3.20e-03],
                [3.52e-02, 3.11e-02, 4.00e-04, -7.70e-03, -3.20e-03, 2.97e-01],
            ]
        ),
        "c_f": 0.22,
        "k": 0.5,
        "nu_f": 0.22,
        "nu_m": 0.385,
    }

    inclusion = mechkit.material.Isotropic(E=inp["E_f"], nu=inp["nu_f"])

    matrix = mechkit.material.Isotropic(E=inp["E_m"], nu=inp["nu_m"])

    averager = mechmean.orientation_averager.AdvaniTucker(N4=inp["N4"])

    for index in range(2):

        ###########################
        # MTOA

        if index == 0:
            P_func = mechmean.hill_polarization.Factory().needle

            input_dict = {
                "phases": {
                    "inclusion": {
                        "material": inclusion,
                        "volume_fraction": inp["c_f"],
                        "hill_polarization": P_func(matrix=matrix),
                    },
                    "matrix": {"material": matrix},
                },
                "k": inp["k"],
                "averaging_func": averager.average,
            }

            mori = mechmean.approximation.MoriTanakaOrientationAveraged(**input_dict)

            C_eff = mori.calc_C_eff()

        ###########################
        # Kehrer

        if index == 1:
            input_dict = {
                "phases": {
                    "inclusion": {
                        "material": inclusion,
                        "volume_fraction": inp["c_f"],
                    },
                    "matrix": {
                        "material": matrix,
                        "volume_fraction": 1.0 - inp["c_f"],
                    },
                },
                "k": inp["k"],
                "averaging_func": averager.average,
            }

            hashin = mechmean.approximation.Kehrer2019(**input_dict)

            C_eff = hashin.calc_C_eff()

        printQueue = [
            "C_eff",
        ]

        # Print
        for val in printQueue:
            print(val)
            pprint.pprint(eval(val))
            print()
