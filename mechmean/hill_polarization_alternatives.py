#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polarizations for isotropic matrices
"""

import numpy as np
from scipy.spatial.transform import Rotation

import mechkit

TOLERANCE_SPHERE = 1e-4


class PConverter(object):
    def __init__(self):
        self.con = mechkit.notation.Converter()

    def eshelby_to_hill(self, E, matrix):
        E_m = self.con.to_mandel6(E)
        C_m = self.con.to_mandel6(matrix.stiffness)
        P = E_m @ np.linalg.inv(C_m)
        return P


class Mura(object):
    """Hill polarization tensor following [Mura1987]_.
    Detailed references are available in source code docstrings"""

    def __init__(self):
        self.tolerance = TOLERANCE_SPHERE

    def spheroid(self, **kwargs):
        """First axis is aligned in e_0 direction"""

        E = self._spheroid_eshelby(**kwargs)
        # Convert Eshelby to Hill
        H = PConverter().eshelby_to_hill(E=E, matrix=kwargs["matrix"])
        return H

    def _spheroid_eshelby(self, aspect_ratio, matrix):
        """Calc Eshelby polarization tensor
        First axis is aligned in e_0 direction
        """

        axes = self._axes_by_aspect_ratio(aspect_ratio=aspect_ratio)

        function_I = self._get_function_I(aspect_ratio=aspect_ratio)

        integral = function_I(axes=axes)

        E = self._eshelby_by_integrals(
            a=axes, I=integral, poisson_matrix=matrix.poisson
        )
        return E

    def _int_index_by_str_add(self, *args):
        """String-add arguments to integer index"""
        return int("".join(map(str, args)))

    def _axes_by_aspect_ratio(self, aspect_ratio):
        """Calc principal half axes of spheroid with a[1] != a[2] == a[3] == 1.0

        Returns
        -------
        dict
            Principal half axes.
        """

        a = {}
        a[2] = a[3] = 1.0
        a[1] = aspect_ratio
        return a

    def _get_function_I(self, aspect_ratio):
        """Select function by aspect_ratio"""

        tol = self.tolerance

        if aspect_ratio <= 1.0 - tol:
            f = self._I_oblate_spheroid
        elif (1.0 - tol < aspect_ratio) and (aspect_ratio < 1.0 + tol):
            f = self._I_sphere
        else:
            f = self._I_prolate_spheroid
        return f

    def _I_sphere(self, axes):
        """Calc integrals for sphere following [Mura1987]_ page 79, (11.21)

        Parameters
        ----------
        axes : dict
            Principal half axes.
            Required keys are [1, 2, 3].

        Returns
        -------
        dict
            Keys are [1, 2, 3, 11, 22, 33, 12, 21, 23, 32, 31, 13].
        """

        a = axes

        tol = self.tolerance
        assert all(
            [
                1.0 - tol < a[1] / a[2],
                a[1] / a[2] < 1.0 + tol,
                1.0 - tol < a[1] / a[3],
                a[1] / a[3] < 1.0 + tol,
            ]
        ), "Aspect ratio of sphere must be 1.0"

        r = a[1]

        I = {}
        I[1] = I[2] = I[3] = 4.0 * np.pi / 3.0
        I[11] = I[22] = I[33] = I[12] = I[23] = I[31] = 4.0 * np.pi / (5.0 * r * r)

        I[21] = I[12]
        I[13] = I[31]
        I[32] = I[23]
        return I

    def _I_oblate_spheroid(self, axes):
        """Calc integrals of oblate spheroid [Mura1987]_ page 84, (11.28)

        Parameters
        ----------
        axes : dict
            Principal half axes.
            Required keys are [1, 2, 3].

        Returns
        -------
        dict
            Keys are [1, 2, 3, 11, 22, 33, 12, 21, 23, 32, 31, 13].
        """

        a = axes

        assert all(
            [a[1] < a[2], a[2] == a[3]]
        ), "Aspect ratio of oblate spheroid must be less than 1.0"

        I = {}
        I[3] = I[2] = (
            (2.0 * np.pi * a[3] * a[3] * a[1])
            / (np.power(a[3] * a[3] - a[1] * a[1], 3.0 / 2.0))
            * (
                np.arccos(a[1] / a[3])
                - a[1] / a[3] * np.power(1.0 - a[1] * a[1] / (a[3] * a[3]), 0.5)
            )
        )

        I[1] = 4.0 * np.pi - 2.0 * I[3]

        I[33] = I[22] = I[32] = np.pi / (a[3] * a[3]) - (I[3] - I[1]) / (
            4.0 * (a[1] * a[1] - a[3] * a[3])
        )

        I[31] = I[21] = (I[3] - I[1]) / (a[1] * a[1] - a[3] * a[3])

        I[11] = (4.0 * np.pi) / (3.0 * a[1] * a[1]) - 2.0 / 3.0 * I[31]

        I[12] = I[21]
        I[13] = I[31]
        I[23] = I[32]
        return I

    def _I_prolate_spheroid(self, axes):
        """Calc integrals of prolate spheroid [Mura1987]_, page 84, (11.29)

        Parameters
        ----------
        axes : dict
            Principal half axes.
            Required keys are [1, 2, 3].

        Returns
        -------
        dict
            Keys are [1, 2, 3, 11, 22, 33, 12, 21, 23, 32, 31, 13].
        """

        a = axes

        assert all(
            [a[1] > a[2], a[2] == a[3]]
        ), "Aspect ratio of prolate spheroid must be greater than 1.0"

        I = {}
        I[2] = I[3] = (
            (2.0 * np.pi * a[1] * a[3] * a[3])
            / (np.power(a[1] * a[1] - a[3] * a[3], 3.0 / 2.0))
            * (
                a[1] / a[3] * np.power(a[1] * a[1] / (a[3] * a[3]) - 1.0, 0.5)
                - np.arccosh(a[1] / a[3])
            )
        )

        I[1] = 4.0 * np.pi - 2.0 * I[2]

        I[12] = I[13] = (I[2] - I[1]) / (a[1] * a[1] - a[2] * a[2])

        I[11] = (4.0 * np.pi) / (3.0 * a[1] * a[1]) - 2.0 / 3.0 * I[12]

        I[22] = I[33] = I[23] = np.pi / (a[2] * a[2]) - (I[2] - I[1]) / (
            4.0 * (a[1] * a[1] - a[2] * a[2])
        )

        I[21] = I[12]
        I[31] = I[13]
        I[32] = I[23]
        return I

    def _eshelby_by_integrals(self, a, I, poisson_matrix):
        """Create Eshelby polarization tensor from non-vanishing integrals
        [Mura1987]_, page 77

        Parameters
        ----------
        a : dict
            Principal half axes.
            Required keys are [1, 2, 3].

        I : dict
            Required keys are [1, 2, 3, 11, 22, 33, 12, 21, 23, 32, 31, 13].

        poisson_matrix : float
            Poisson ratio of the matrix.

        Returns
        -------
        np.array
            Eshelby polarization tensor.
            Symmetries: Minor left and minor right. No major symmetry.
        """

        # Aliases
        add = self._int_index_by_str_add

        # Calc recurring terms once
        t1 = 8.0 * np.pi * (1.0 - poisson_matrix)
        t2 = 1.0 - 2.0 * poisson_matrix
        t3 = t2 / t1

        # Start with zeros
        E = np.zeros((3, 3, 3, 3), dtype="float64")

        simultaneous_cyclic_permutations = [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ]

        # Define components of polarization tensor by integrals
        for k, l, m in simultaneous_cyclic_permutations:
            k0, l0, m0 = k - 1, l - 1, m - 1
            E[k0, k0, k0, k0] = 3.0 / t1 * a[k] * a[k] * I[add(k, k)] + t3 * I[k]
            E[k0, k0, l0, l0] = 1.0 / t1 * a[l] * a[l] * I[add(k, l)] - t3 * I[k]
            E[k0, k0, m0, m0] = 1.0 / t1 * a[m] * a[m] * I[add(k, m)] - t3 * I[k]
            E[k0, l0, k0, l0] = E[l0, k0, k0, l0] = E[k0, l0, l0, k0] = E[
                l0, k0, l0, k0
            ] = (a[k] * a[k] + a[l] * a[l]) / 2.0 * 1.0 / t1 * I[
                add(k, l)
            ] + t3 / 2.0 * (
                I[k] + I[l]
            )
        return E


class Ortolano(object):
    """Hill polarization tensor [Ortolano2013]_, [Friebel2007]_"""

    def __init__(self):
        self.con = mechkit.notation.Converter()

    def needle(self, matrix):
        """Spheroid with an infite aspect_ratio

        Cylinders of original formulas point into z-direction.
        As the x-direction is the common symmetry axis,
        Eshelbys tensor is rotated to point into x-direction.
        """
        nu = matrix.poisson

        S = np.zeros((3, 3, 3, 3), dtype="float64")
        term1 = 8.0 * (1.0 - nu)

        S[0, 0, 0, 0] = S[1, 1, 1, 1] = (5.0 - 4.0 * nu) / term1
        S[0, 0, 1, 1] = S[1, 1, 0, 0] = (4.0 * nu - 1.0) / term1
        S[0, 0, 2, 2] = S[1, 1, 2, 2] = nu / (2.0 * (1.0 - nu))
        # Bugfix in next line: S[0, 1, 1, 0] instead of two times S[1, 0, 0, 1]
        S[0, 1, 0, 1] = S[1, 0, 0, 1] = S[0, 1, 1, 0] = S[1, 0, 1, 0] = (
            3.0 - 4.0 * nu
        ) / term1
        S[1, 2, 1, 2] = S[2, 1, 1, 2] = S[1, 2, 2, 1] = S[2, 1, 2, 1] = 1.0 / 4.0
        S[2, 0, 2, 0] = S[0, 2, 2, 0] = S[2, 0, 0, 2] = S[0, 2, 0, 2] = 1.0 / 4.0

        # Rotate from z-direction into x-direction
        Q = Rotation.from_rotvec(np.pi / 2 * np.array([0.0, 1.0, 0.0])).as_matrix()
        S_rotated = np.einsum("ij, kl, mn, pq, jlnq -> ikmp", Q, Q, Q, Q, S)

        H = PConverter().eshelby_to_hill(E=S_rotated, matrix=matrix)
        return H
