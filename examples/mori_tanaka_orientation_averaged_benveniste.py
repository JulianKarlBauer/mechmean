#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import mechmean
import mechkit
from mechmean.example_input import inp

# Define isotropic constituents
inclusion = mechkit.material.Isotropic(E=inp["E_f"], nu=inp["nu_f"])
matrix = mechkit.material.Isotropic(E=inp["E_m"], nu=inp["nu_m"])

# Define orientation averager and polairzation
averager = mechmean.orientation_averager.AdvaniTucker(N4=inp["N4"])
P_func = mechmean.hill_polarization.Factory().needle

# Homogenize
input_dict = {
    "phases": {
        "inclusion": {
            "material": inclusion,
            "volume_fraction": inp["c_f"],
            "hill_polarization": P_func(matrix=matrix),
        },
        "matrix": {"material": matrix},
    },
    "averaging_func": averager.average,
}
mori = mechmean.approximation.MoriTanakaOrientationAveragedBenveniste(**input_dict)
C_eff = mori.calc_C_eff()

print("Effective stiffness Mori-Tanaka orientation averaged Benveniste", C_eff)
