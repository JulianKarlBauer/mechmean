#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approximations of effective stiffness of two-phase materials
"""

import numpy as np
import mechmean
import mechkit
import functools
import copy
from collections import namedtuple
from numpy.linalg import inv
from mechmean import utils
import pprint


#####################################################################################
class TwoPhaseComposite(object):
    """See approximation_docstrings for docstrings"""

    def __init__(self):
        self.notation = "mandel6"
        self._I4s = mechkit.notation.Converter().to_mandel6(mechkit.tensors.Basic().I4s)

    def calc_A_SI_from_hill_polarization(self, P_i, C_i, C_m):
        return inv(P_i @ (C_i - C_m) + self._I4s)

    def calc_C_eff_by_A_i(self, c_i, A_i, C_i, C_m):
        return C_m + c_i * ((C_i - C_m) @ A_i)


#####################################################################################
class MoriTanaka(TwoPhaseComposite):
    """See approximation_docstrings for docstrings"""

    def __init__(self, phases, **kwargs):
        self._C_i = phases["inclusion"]["material"].stiffness_mandel6
        self._c_i = phases["inclusion"]["volume_fraction"]
        self._P_i = phases["inclusion"]["hill_polarization"]
        self._C_m = phases["matrix"]["material"].stiffness_mandel6
        super().__init__()

    def calc_A_MT_i(self, c_i, A_SI_i):
        return inv((1.0 - c_i) * inv(A_SI_i) + c_i * self._I4s)

    def calc_C_eff(self):
        self.A_MT_i = self.calc_A_MT_i(c_i=self._c_i, A_SI_i=self.calc_A_SI_i())
        self.C_eff = self.calc_C_eff_by_A_i(
            c_i=self._c_i, A_i=self.A_MT_i, C_i=self._C_i, C_m=self._C_m
        )
        return self.C_eff

    def calc_A_SI_i(self):
        r"""Wrap
        :any:`mechmean.approximation.TwoPhaseComposite.calc_A_SI_from_hill_polarization`
        """
        return self.calc_A_SI_from_hill_polarization(
            P_i=self._P_i, C_i=self._C_i, C_m=self._C_m
        )


class MoriTanakaOrientationAveragedBenveniste(MoriTanaka):
    """See approximation_docstrings for docstrings"""

    def __init__(self, **kwargs):

        self.averaging_func = kwargs["averaging_func"]
        super().__init__(**kwargs)
        self._dC = self._C_i - self._C_m

    def calc_A_SI_i(self):

        A_SI = self.calc_A_SI_from_hill_polarization(
            P_i=self._P_i, C_i=self._C_i, C_m=self._C_m
        )
        A_star = self._dC @ A_SI
        A_star_averaged = self.averaging_func(A_star)
        return inv(self._dC) @ A_star_averaged


class Kehrer2019(object):
    """See approximation_docstrings for docstrings"""

    def __init__(self, phases, averaging_func, k):
        self.phases = phases
        self.averaging_func = averaging_func
        self.k = k

        self.Pud_func = mechmean.hill_polarization.Castaneda().needle

        self.P0_func = functools.partial(
            mechmean.hill_polarization.Castaneda().spheroid, aspect_ratio=1.0
        )

        self._I4s = mechkit.notation.Converter().to_mandel6(mechkit.tensors.Basic().I4s)

    def calc_C_eff(self):
        av = self.averaging_func
        Is = self._I4s

        k = self.k

        mat_i = self.phases["inclusion"]["material"]
        mat_m = self.phases["matrix"]["material"]

        Ci = mat_i.stiffness_mandel6
        Cm = mat_m.stiffness_mandel6
        dC = Cm - Ci

        ci = self.phases["inclusion"]["volume_fraction"]
        cm = 1.0 - ci

        #############################################################
        # Step 1

        P_ud_i = self.Pud_func(matrix=mat_i)
        P_ud_m = self.Pud_func(matrix=mat_m)

        tmp_upp = inv(Is + ci * P_ud_i @ dC)
        tmp_low = inv(Is - cm * P_ud_m @ dC)

        C_upp = Ci + cm * dC @ tmp_upp
        C_low = Cm - ci * dC @ tmp_low

        self._C_ud_upp = C_upp
        self._C_ud_low = C_low

        #############################################################
        # Step 2

        ref_mat = (1.0 - k) * mat_m + k * mat_i

        self._C0 = C0 = ref_mat.stiffness_mandel6

        P0 = self.P0_func(matrix=ref_mat)
        P0inv = inv(P0)

        Astar_upp = inv(P0inv + C_upp - C0)
        Astar_low = inv(P0inv + C_low - C0)

        self.C_eff_upp = C0 - P0inv + inv(av(Astar_upp))
        self.C_eff_low = C0 - P0inv + inv(av(Astar_low))

        NT = namedtuple("Effective_stiffness", ["upper", "lower"])
        return NT(self.C_eff_upp, self.C_eff_low)


class HashinShtrikmanWalpole(object):
    def __init__(self, phases, P_func, averaging_func):
        self.P_func = P_func
        for key, phase in phases.items():
            phases[key]["stiffness"] = phase["material"].stiffness_mandel6
        self.phases = phases
        self.averaging_func = averaging_func

    def calc_C_eff(self, ref_material):
        P = self._calc_P(ref_material=ref_material)

        func = self._select_func_depending_on_P(P)

        return func(
            phases=copy.deepcopy(self.phases), P=P, C_0=ref_material.stiffness_mandel6
        )

    def _calc_P(self, ref_material):
        return self.P_func(matrix=ref_material)

    def _select_func_depending_on_P(self, P):
        invertible = mechmean.utils.isinvertible(P)
        funcs = {
            True: self._calc_C_eff_for_invertible_P,
            False: self._calc_C_eff_for_singular_P_using_dual_scheme,
        }
        return funcs[invertible]

    def _calc_C_eff_for_invertible_P(self, phases, P, C_0):
        P_inv = inv(P)

        for key, phase in phases.items():
            phases[key]["W"] = inv(phase["stiffness"] - C_0 + P_inv)

        W_av = self.averaging_func(phases=phases)

        return C_0 - P_inv + inv(W_av)

    def _calc_C_eff_for_singular_P_using_dual_scheme(self, phases, P, C_0):
        r"""Handle singular P using dual scheme following
        [Walpole1969]_ (page 238).
        """
        # Calc inverse Polarization and compliance
        con = mechkit.notation.Converter()
        I4s = con.to_mandel6(mechkit.tensors.Basic().I4s)
        Q = C_0 @ (I4s - P @ C_0)

        S_0 = inv(C_0)

        # Invert stiffnesses of phases
        for key, phase in phases.items():
            phases[key]["stiffness"] = inv(phase["stiffness"])

        S_eff = self._calc_C_eff_for_invertible_P(phases=phases, P=Q, C_0=S_0)
        return inv(S_eff)

    def calc_strain_localization_tensor(self, key_phase, ref_material):
        P = self._calc_P(ref_material=ref_material)
        C_0 = ref_material.stiffness_mandel6
        phases = copy.deepcopy(self.phases)

        func_A_SI = TwoPhaseComposite().calc_A_SI_from_hill_polarization

        for key, phase in phases.items():
            phases[key]["A_SI"] = func_A_SI(P_i=P, C_i=phase["stiffness"], C_m=C_0)

        A_SI_average = self.averaging_func(phases=phases, key_to_be_averaged="A_SI")

        return phases[key_phase]["A_SI"] @ inv(A_SI_average)

    def calc_stress_localization_tensor(self, key_phase, ref_material):
        kwargs = locals()
        A = self.calc_strain_localization_tensor(**kwargs)
        S_0 = inv(ref_material.stiffness_mandel6)
        return self.phases[key_phase]["stiffness"] @ A @ S_0

    def localize_strain(self, key_phase, ref_material, strain_macro):
        kwargs = locals()
        A = self.calc_strain_localization_tensor(**kwargs)
        return A @ strain_macro

    def localize_stress(self, key_phase, ref_material, stress_macro):
        kwargs = locals()
        B = self.calc_stress_localization_tensor(**kwargs)
        return B @ stress_macro


class HSW_VolumeFraction(HashinShtrikmanWalpole):
    """See approximation_docstrings for docstrings"""

    class AveragerKeywordWeightedSum(object):
        def __call__(
            self, phases, key_to_be_averaged="W", key_weight="volume_fraction"
        ):
            sum = 0.0
            for _, phase in phases.items():
                sum += phase[key_weight] * phase[key_to_be_averaged]
            return sum

    def __init__(self, phases, P_func):
        super().__init__(
            phases=phases,
            P_func=P_func,
            averaging_func=self.AveragerKeywordWeightedSum(),
        )


class HSW_SinglePhaseOrientation(HashinShtrikmanWalpole):
    """See approximation_docstrings for docstrings"""

    class AveragerOrientationWithKeyword(object):
        def __init__(self, averager):
            self.averager = averager

        def __call__(self, phases, key_to_be_averaged="W"):
            if len(phases) != 1:
                raise utils.Ex(
                    "More than one phase."
                    "Please use HashinShtrikmanWalpole instead of"
                    "HashinShtrikmanWalpole_SinglePhaseOrientation"
                )

            key_phase = list(phases.keys())[0]
            return self.averager(phases[key_phase][key_to_be_averaged])

    def __init__(self, phases, P_func, orientation_averager):
        super().__init__(
            phases=phases,
            P_func=P_func,
            averaging_func=self.AveragerOrientationWithKeyword(
                averager=orientation_averager,
            ),
        )


class Kehrer2019HSW(object):
    """See approximation_docstrings for docstrings"""

    def __init__(self, phases, averaging_func, k):
        self.inclusion = phases["inclusion"]["material"]
        self.volume_fraction = phases["inclusion"]["volume_fraction"]
        self.matrix = phases["matrix"]["material"]
        self.averaging_func = averaging_func
        self.k = k

        self.Pud_func = mechmean.hill_polarization.Castaneda().needle

        self.P0_func = functools.partial(
            mechmean.hill_polarization.Castaneda().spheroid, aspect_ratio=1.0
        )

    def calc_C_eff(self):

        # Step 1
        hashin1 = HSW_VolumeFraction(
            phases={
                "i": {
                    "material": self.inclusion,
                    "volume_fraction": self.volume_fraction,
                },
                "m": {
                    "material": self.matrix,
                    "volume_fraction": 1.0 - self.volume_fraction,
                },
            },
            P_func=self.Pud_func,
        )

        self.C_ud_upp = hashin1.calc_C_eff(ref_material=self.inclusion)
        self.C_ud_low = hashin1.calc_C_eff(ref_material=self.matrix)

        # Step 2
        hashin2_upp = self._get_hashin2(stiffness=self.C_ud_upp)
        hashin2_low = self._get_hashin2(stiffness=self.C_ud_low)

        k = self.k

        ref_mat = (1.0 - k) * self.matrix + k * self.inclusion

        self.C_eff_upp = hashin2_upp.calc_C_eff(ref_material=ref_mat)
        self.C_eff_low = hashin2_low.calc_C_eff(ref_material=ref_mat)

        NT = namedtuple("Effective_stiffness", ["upper", "lower"])
        return NT(self.C_eff_upp, self.C_eff_low)

    def _get_hashin2(self, stiffness):
        # ToDo: Fix this Quickfix
        class Bunch(object):
            pass

        instance = Bunch()
        instance.stiffness_mandel6 = stiffness

        return HSW_SinglePhaseOrientation(
            phases={
                "UD_material": {"material": instance},
            },
            P_func=self.P0_func,
            orientation_averager=self.averaging_func,
        )


class HSW2StepInterpolatedReferenceMaterial:
    """See approximation_docstrings for docstrings"""

    def __init__(self, phases, averaging_func, k1, k2):
        self.inclusion = phases["inclusion"]["material"]
        self.volume_fraction = phases["inclusion"]["volume_fraction"]
        self.matrix = phases["matrix"]["material"]
        self.averaging_func = averaging_func
        self.k1 = k1
        self.k2 = k2

        self.Pud_func = mechmean.hill_polarization.Castaneda().needle

        self.P0_func = functools.partial(
            mechmean.hill_polarization.Castaneda().spheroid, aspect_ratio=1.0
        )

    def calc_C_eff(self):

        # Step 1
        hashin_1 = HSW_VolumeFraction(
            phases={
                "i": {
                    "material": self.inclusion,
                    "volume_fraction": self.volume_fraction,
                },
                "m": {
                    "material": self.matrix,
                    "volume_fraction": 1.0 - self.volume_fraction,
                },
            },
            P_func=self.Pud_func,
        )

        self.C_ud = hashin_1.calc_C_eff(
            ref_material=self.interpolate_reference_material(k=self.k1)
        )

        # Step 2
        hashin2 = HSW_SinglePhaseOrientation(
            phases={
                "UD_material": {
                    "material": self.make_stiffness_attribute_of_object(
                        stiffness=self.C_ud
                    ),
                },
            },
            P_func=self.P0_func,
            orientation_averager=self.averaging_func,
        )

        self.C_eff = hashin2.calc_C_eff(
            ref_material=self.interpolate_reference_material(k=self.k2)
        )

        return self.C_eff

    def interpolate_reference_material(self, k):
        return (1.0 - k) * self.matrix + k * self.inclusion

    def make_stiffness_attribute_of_object(self, stiffness):
        # ToDo: Fix this Quickfix
        class Bunch(object):
            pass

        instance = Bunch()
        instance.stiffness_mandel6 = stiffness

        return instance


