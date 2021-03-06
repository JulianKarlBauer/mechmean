# # Two-step Hashin-Shtrikman interpolated twice

import numpy as np
import mechmean
import mechkit
from mechmean.example_input import inp

np.set_printoptions(linewidth=140)

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
        },
        "matrix": {
            "material": matrix,
            "volume_fraction": 1.0 - inp["c_f"],
        },
    },
    "k1": 1.0 / 2.0,
    "k2": 1.0 / 2.0,
    "averaging_func": averager.average,
}
hashin = mechmean.approximation.HSW2StepInterpolatedReferenceMaterial(**input_dict)
C_eff = hashin.calc_C_eff()

print("Effective stiffness Hashin Shtrikman two step with two interpolations")
print(C_eff)
