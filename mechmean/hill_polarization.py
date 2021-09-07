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
    """Hill polarization tensor"""

    def __init__(self):
        self.notation = "mandel6"

    def spheroid_mura(self, aspect_ratio, matrix):
        """Detailed docstring"""
        kwargs = locals()
        kwargs.pop("self")
        return Mura().spheroid(**kwargs)

    def spheroid_castaneda(self, aspect_ratio, matrix):
        """Detailed docstring"""
        kwargs = locals()
        kwargs.pop("self")
        return Castaneda().spheroid(**kwargs)


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

    def spheroid(
        self,
        aspect_ratio,
        matrix,
    ):
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


class Mura(object):
    """Hill polarization tensor [Mura1987]_

    References
    ----------

    .. [Mura1987] Mura, T., Micromechanics of Defects in Solids
        (Martinus Nijhoff, Dordrecht, 1987). CrossRef zbMATH.

    """

    def __init__(self):
        self.tolerance = TOLERANCE_SPHERE

    def spheroid(self, **kwargs):
        """Convert Eshelby to Hill"""

        E = self.spheroid_eshelby(**kwargs)
        H = PConverter().eshelby_to_hill(E=E, matrix=kwargs["matrix"])
        return H

    def spheroid_eshelby(self, aspect_ratio, matrix):
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
        """Calc integrals for sphere following [Mura1987, page 79, (11.21)]

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
        """Calc integrals of oblate spheroid [Mura1987, page 84, (11.28)]

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
        """Calc integrals of prolate spheroid [Mura1987, page 84, (11.29)]

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
    """Hill polarization tensor [Ortolano2013]_, [Friebel2007]_

    References
    ----------

    .. [Ortolano2013] Ortolano González, J.M., Hernández Ortega, J.A. and
        Oliver Olivella, X., 2013. A comparative study on homogenization
        strategies for multi-scale analysis of materials. Centre
        Internacional de Mètodes Numèrics en Enginyeria (CIMNE).

    .. [Friebel2007] Friebel, C., 2007. Mechanics and Acoustics of
        viscoelastic inclusion reinforced composites: micro-macro modeling
        of effective properties (Doctoral dissertation, PhD thesis,
        Université catholique de Louvain, Belgium, 2007. 42).

    """

    def __init__(self):
        self.con = mechkit.notation.Converter()

    def needle(self, matrix):
        """Spheroid with aspect_ratio == infinity

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


if __name__ == "__main__":
    import mechmean

    np.set_printoptions(
        linewidth=140,
        precision=2,
        # suppress=False,
    )

    factory = mechmean.hill_polarization.Factory()

    for K, G in [
        (1e6, 4e5),
    ]:
        matrix = mechkit.material.Isotropic(K=K, G=G)
        for aspect_ratio in [1.0]:

            P_Spheroid = mechmean.hill_polarization.Factory().spheroid_mura(
                aspect_ratio=50, matrix=matrix
            )

            P_Kehrer = mechmean.hill_polarization.Castaneda().needle(matrix=matrix)

            P_Ortolano = mechmean.hill_polarization.Ortolano().needle(matrix=matrix)

            printQueue = [
                "P_Spheroid",
                "P_Kehrer",
                "P_Ortolano",
                "np.allclose(P_Ortolano, P_Kehrer)",
                "np.allclose(P_Spheroid, P_Kehrer)",
            ]
            # Print
            for val in printQueue:
                print(val)
                print(eval(val), "\n")
