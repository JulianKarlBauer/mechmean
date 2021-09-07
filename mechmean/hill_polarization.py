#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polarizations for isotropic matrices
"""

import numpy as np
from scipy.spatial.transform import Rotation

from mechmean import utils
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


class Factory(object):
    """Factory layer for Hill polarization tensors"""

    def spheroid(self, aspect_ratio, matrix):
        """Detailed docstring"""
        return Castaneda().spheroid(aspect_ratio=aspect_ratio, matrix=matrix)


class Castaneda(object):
    r"""Hill polarization tensor

    References
    ----------

    .. [Castaneda1997] Castaneda, P.P. and Suquet, P., 1997. Nonlinear
        composites. In Advances in applied mechanics
        (Vol. 34, pp. 171-302). Elsevier.

    .. [Willis1981] Willis, J.R., 1981. Variational and related methods
        for the overall properties of composites. In Advances in applied
        mechanics (Vol. 21, pp. 1-78). Elsevier.

    .. [Brylka2017] Brylka, B., Charakterisierung und Modellierung der
        Steifigkeit von langfaserverstärktem Polypropylen. 10, (2017).

    .. [Kehrer2019] Kehrer, M.L., 2019. Thermomechanical Mean-Field
        Modeling and Experimental Characterization of Long Fiber-Reinforced
        Sheet Molding Compound Composites (Vol. 15).
        KIT Scientific Publishing.

    """

    def __init__(self):
        self.tolerance = TOLERANCE_SPHERE

        self.con = mechkit.notation.Converter()
        self.tensors = mechkit.tensors.Basic()

    def needle(self, matrix):
        r"""[Kehrer2019]_ formula (A.7)"""
        la = matrix.la
        mu = matrix.G

        k = 1 / 4.0 * 1.0 / (la + 2.0 * mu)
        m = 0.0
        r = 0.0
        p = 1 / 2.0 * (la + 3.0 * mu) / (4.0 * (la + 2.0 * mu) * mu)
        q = 1 / 2.0 * 1.0 / (4.0 * mu)

        return self._P_by_kmpqr(k=k, m=m, p=p, q=q, r=r)

    def spheroid(self, aspect_ratio, matrix):
        """Calc Hills polarization tensor for spheroid combining
        appendix C of [Castaneda1997]_,
        equation (4.39) [Willis1981]_,
        formulas (2.57) and (2.58) in [Brylka2017]_.

        The first principle semi-axis is aligned with the global axis e_0.
        """

        kwargs = locals()
        kwargs.pop("self")

        func = self._get_func_oblate_prolate_or_sphere(aspect_ratio=aspect_ratio)
        return func(**kwargs)

    def _get_func_oblate_prolate_or_sphere(self, aspect_ratio):

        tol = self.tolerance

        if aspect_ratio <= 1.0 - tol:
            f = self._oblate_or_prolate
        elif (1.0 - tol < aspect_ratio) and (aspect_ratio < 1.0 + tol):
            f = self._sphere
        else:
            f = self._oblate_or_prolate
        return f

    def _oblate_or_prolate(self, aspect_ratio, matrix):
        def h(a):
            """Temporary parameter depending on aspect ratio"""
            a2 = a * a
            if a < 1.0:
                h = (a * (np.arccos(a) - a * np.sqrt(1.0 - a2))) / np.power(
                    1.0 - a2, 3.0 / 2.0
                )
            elif a == 1.0:
                raise utils.Ex("Aspect ratio = 1.0 not supported")
            else:
                h = (a * (a * np.sqrt(a2 - 1.0) - np.arccosh(a))) / np.power(
                    a2 - 1.0, 3.0 / 2.0
                )
            return h

        # Abbreviations
        a = aspect_ratio
        K = matrix.K
        G = matrix.G

        a2 = a * a
        term = (1.0 - a2) * G * (4.0 * G + 3.0 * K)

        # Calc
        h = h(a)

        m = ((G + 3.0 * K) * (2.0 * a2 - h - 2.0 * a2 * h)) / (4.0 * term)

        k = (
            G * (7.0 * h - 2.0 * a2 - 4.0 * a2 * h)
            + 3.0 * K * (h - 2.0 * a2 + 2.0 * a2 * h)
        ) / (8.0 * term)

        r = (
            G * (6.0 - 5.0 * h - 8.0 * a2 + 8.0 * a2 * h)
            + 3.0 * K * (h - 2.0 * a2 + 2.0 * a2 * h)
        ) / (2.0 * term)

        p = (
            G * (15.0 * h - 2.0 * a2 - 12.0 * a2 * h) + 3.0 * K * (3.0 * h - 2.0 * a2)
        ) / (16.0 * term)

        q = (
            2.0 * G * (4.0 - 3.0 * h - 2.0 * a2)
            + 3.0 * K * (2.0 - 3.0 * h + 2.0 * a2 - 3.0 * a2 * h)
        ) / (8.0 * term)

        return self._P_by_kmpqr(k=k, m=m, p=p, q=q, r=r)

    def _sphere(self, matrix, **kwargs):
        """[Willis1981]_ eq. (4.39)"""
        K = matrix.K
        G = matrix.G

        alpha = 1.0 / (3.0 * K + 4.0 * G)
        beta = (3.0 * (K + 2.0 * G)) / (5.0 * G * (3.0 * K + 4.0 * G))

        H = alpha * self.tensors.P1 + beta * self.tensors.P2

        return self.con.to_mandel6(H)

    def _P_by_kmpqr(self, k, m, p, q, r):
        P = np.array(
            [
                [r, m, m, 0.0, 0.0, 0.0],
                [m, k + p, k - p, 0.0, 0.0, 0.0],
                [m, k - p, k + p, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0 * p, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0 * q, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 * q],
            ],
            dtype="float64",
        )
        return P
