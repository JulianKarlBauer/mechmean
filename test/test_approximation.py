#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import mechkit
import numpy as np
import pprint
from itertools import groupby
import pytest
import functools

import mechmean

np.set_printoptions(linewidth=140, precision=2)

#########################################
# Fixtures and data


def run_test_multiple_times(func, nbr=10):
    inps = []
    inps.append(create_standard_input())
    for i in range(10):
        inps.append(create_random_inp())

    for inp in inps:
        pprint.pprint(inp)
        func(inp)


def create_random_inp():

    bounds = {
        "E_f": {"low": 10.0, "up": 1000},
        "E_m": {"low": 1.0, "up": 1000},
        "c_f": {"low": 0.0, "up": 1},
        "k": {"low": 0.0, "up": 1},
        "nu_f": {"low": 0.0, "up": 0.5},
        "nu_m": {"low": 0.0, "up": 0.5},
    }
    inp = {}
    for key, bound in bounds.items():
        inp[key] = np.random.uniform(bound["low"], bound["up"])

    inp["N4"] = N4_from_orientations(create_random_orientations())
    add_inclusion_matrix(inp)
    return inp


def add_inclusion_matrix(inp):
    inp["inclusion"] = mechkit.material.Isotropic(E=inp["E_f"], nu=inp["nu_f"])
    inp["matrix"] = mechkit.material.Isotropic(E=inp["E_m"], nu=inp["nu_m"])
    return inp


def create_random_orientations():
    bounds_nbr_orientations = {"low": 1.0, "up": 100}
    bounds_value_orientations = {"low": 0.0, "up": 1.0}

    b_nbr = bounds_nbr_orientations
    b_val = bounds_value_orientations

    orientations = []
    for i in range(int(np.random.uniform(b_nbr["low"], b_nbr["up"]))):
        orientations.append(
            [np.random.uniform(b_val["low"], b_val["up"]) for i in range(3)]
        )
    return orientations


def N4_from_orientations(orientations):
    N4 = mechmean.fabric_tensors.first_kind_discrete(order=4, orientations=orientations)
    return N4


def create_standard_input():
    inp = {
        "E_f": 73.0,
        "E_m": 3.4,
        "c_f": 0.22,
        "k": 1.0,
        "nu_f": 0.22,
        "nu_m": 0.385,
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
    }
    add_inclusion_matrix(inp)
    return inp


def input_dict_from_input(inp, averager):
    input_dict = {
        "phases": {
            "inclusion": {
                "material": inp["inclusion"],
                "volume_fraction": inp["c_f"],
            },
            "matrix": {"material": inp["matrix"]},
        },
        "k": inp["k"],
        "averaging_func": averager.average,
    }
    return input_dict


#########################################
# Tests


def test_mori_tanaka_orientation_averaged_comparison_implementations():
    """Successful but private"""
    assert True


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_hs2step_kehrer_compare_implementations_Kehrer2019_Kehrer2019HSW():
    run_test_multiple_times(
        func=hs2step_kehrer_compare_implementations_Kehrer2019_Kehrer2019HSW,
    )


def hs2step_kehrer_compare_implementations_Kehrer2019_Kehrer2019HSW(inp):

    averager = mechmean.orientation_averager.AdvaniTucker(N4=inp["N4"])

    input_dict = input_dict_from_input(inp, averager)

    kehrer = mechmean.approximation.Kehrer2019(**input_dict)

    kehrer.calc_C_eff()

    ###########################

    hashin = mechmean.approximation.Kehrer2019HSW(**input_dict)
    hashin.calc_C_eff()

    ###########################
    # Print

    comparisons = {
        "all": {"slice": np.s_[:, :], "tolerance": {"rtol": 1e-05, "atol": 1e-04}},
    }

    comparisons = {
        "C_ud_low": {
            "pair": [kehrer._C_ud_low, hashin.C_ud_low],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
        "C_ud_upp": {
            "pair": [kehrer._C_ud_upp, hashin.C_ud_upp],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
        "C_eff_upp": {
            "pair": [kehrer.C_eff_upp, hashin.C_eff_upp],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
        "C_eff_low": {
            "pair": [kehrer.C_eff_low, hashin.C_eff_low],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
    }

    for key, comp in comparisons.items():
        kehrer, hashin = comp["pair"]
        printQueue = [
            "key",
            "kehrer-hashin",
            "np.divide(kehrer-hashin, hashin)",
        ]

        # Print
        with np.errstate(divide="ignore"):
            for val in printQueue:
                print(val)
                print(eval(val), "\n")

        assert np.allclose(kehrer, hashin, **comp["tolerance"])


def test_hashin_shtrikman_implementations():
    """Successful but private"""
    assert True


def test_MT_is_HS_with_matrix_as_reference_material():

    inp = create_standard_input()

    add_inclusion_matrix(inp)

    P_func = functools.partial(
        mechmean.hill_polarization.Castaneda().spheroid, aspect_ratio=1.0
    )

    hashin = mechmean.approximation.HSW_VolumeFraction(
        phases={
            "i": {"material": inp["inclusion"], "volume_fraction": inp["c_f"]},
            "m": {"material": inp["matrix"], "volume_fraction": 1.0 - inp["c_f"]},
        },
        P_func=P_func,
    )

    mori = mechmean.approximation.MoriTanaka(
        phases={
            "inclusion": {
                "material": inp["inclusion"],
                "hill_polarization": P_func(matrix=inp["matrix"]),
                "volume_fraction": inp["c_f"],
            },
            "matrix": {"material": inp["matrix"]},
        },
    )

    assert np.allclose(hashin.calc_C_eff(ref_material=inp["matrix"]), mori.calc_C_eff())


def test_HSW_strain_localization_sum_to_one():
    run_test_multiple_times(func=HSW_strain_localization_sum_to_one)


def HSW_strain_localization_sum_to_one(inp):

    input_dict = {
        "phases": {
            "i": {"material": inp["inclusion"], "volume_fraction": inp["c_f"]},
            "m": {"material": inp["matrix"], "volume_fraction": 1.0 - inp["c_f"]},
        },
        "P_func": functools.partial(
            mechmean.hill_polarization.Castaneda().spheroid, aspect_ratio=1.0
        ),
    }

    hashin = mechmean.approximation.HSW_VolumeFraction(**input_dict)

    A = {
        key: hashin.calc_strain_localization_tensor(
            key_phase=key, ref_material=inp["matrix"]
        )
        for key in ["i", "m"]
    }

    A_av = np.zeros((6, 6))
    for key, val in input_dict["phases"].items():
        A_av += val["volume_fraction"] * A[key]

    printQueue = [
        "A",
        "A_av",
    ]
    for val in printQueue:
        print(val)
        pprint.pprint(eval(val))
        print()

    assert np.allclose(A_av, np.eye(6))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compare_implementations_Kehrer2019HSW_HSW2StepInterpolatedReferenceMaterial():
    run_test_multiple_times(
        func=compare_implementations_Kehrer2019HSW_HSW2StepInterpolatedReferenceMaterial,
    )


def compare_implementations_Kehrer2019HSW_HSW2StepInterpolatedReferenceMaterial(inp):

    averager = mechmean.orientation_averager.AdvaniTucker(N4=inp["N4"])

    input_dict = input_dict_from_input(inp, averager)

    kehrerHSW = mechmean.approximation.Kehrer2019HSW(**input_dict)
    kehrerHSW.calc_C_eff()

    k = input_dict.pop("k")
    input_dict["k2"] = k

    generic_upper = mechmean.approximation.HSW2StepInterpolatedReferenceMaterial(
        **input_dict, k1=1.0
    )
    generic_upper.calc_C_eff()

    generic_lower = mechmean.approximation.HSW2StepInterpolatedReferenceMaterial(
        **input_dict, k1=0.0
    )
    generic_lower.calc_C_eff()

    ###########################
    # Print

    comparisons = {
        "all": {
            "slice": np.s_[:, :],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
    }

    comparisons = {
        "C_ud_low": {
            "pair": [kehrerHSW.C_ud_low, generic_lower.C_ud],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
        "C_ud_upp": {
            "pair": [kehrerHSW.C_ud_upp, generic_upper.C_ud],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
        "C_eff_upp": {
            "pair": [kehrerHSW.C_eff_upp, generic_upper.C_eff],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
        "C_eff_low": {
            "pair": [kehrerHSW.C_eff_low, generic_lower.C_eff],
            "tolerance": {"rtol": 1e-05, "atol": 1e-04},
        },
    }

    for key, comp in comparisons.items():
        kehrer, generic = comp["pair"]
        printQueue = [
            "key",
            "kehrer-generic",
            "np.divide(kehrer-generic, generic)",
        ]

        # Print
        with np.errstate(divide="ignore"):
            for val in printQueue:
                print(val)
                print(eval(val), "\n")

        assert np.allclose(
            kehrer,
            generic,
            **comp["tolerance"],
        )


if __name__ == "__main__":
    test_compare_implementations_Kehrer2019HSW_HSW2StepInterpolatedReferenceMaterial()
