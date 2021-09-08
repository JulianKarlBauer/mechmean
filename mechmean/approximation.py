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
    def __init__(self):
        self.notation = "mandel6"
        self._I4s = mechkit.notation.Converter().to_mandel6(mechkit.tensors.Basic().I4s)

    def calc_A_SI_from_hill_polarization(self, P_i, C_i, C_m):
        return inv(P_i @ (C_i - C_m) + self._I4s)

    def calc_C_eff_by_A_i(self, c_i, A_i, C_i, C_m):
        return C_m + c_i * ((C_i - C_m) @ A_i)


TwoPhaseComposite.__doc__ = r"""
    Base class for approximations in two-phase materials.

    Formulations are based on average **strain localization tensors**
    mapping average strain to average strain in phase j,
    i.e.

    .. math::
        \begin{align*}
            \left<\mathbb{\varepsilon}\right>_{\text{j}}
            =
            \mathbb{A}_{\text{j}}
            \left[
            \left<\mathbb{\varepsilon}\right>
            \right]
        \end{align*}
    """

TwoPhaseComposite.calc_A_SI_from_hill_polarization.__doc__ = r"""
        Calc strain localization tensor in single inclusion problem.

       .. math::
           \begin{align*}
               \mathbb{A}_{\text{i}}^{\text{SI}}
               =
               \left(
                   \mathbb{P}_{\text{i}}
                   \left(
                       \mathbb{C}_{\text{i}}
                       - \mathbb{C}_{\text{m}}
                   \right)
                   + \mathbb{I}^{\text{S}}
               \right)^{-1}
           \end{align*}

       Parameters
       ----------
       P_i : np.array (mandel6_4)
           Hill polarization of inclusion.
       C_i : np.array (mandel6_4)
           Stiffness of inclusion.
       C_m : np.array (mandel6_4)
           Stiffness of matrix.

       Returns
       -------
       np.array (mandel6_4)
           Strain localization in single inclusion problem.

       """

TwoPhaseComposite.calc_C_eff_by_A_i.__doc__ = r"""
        Calc effective stiffness of two phase material

        .. math::
            \begin{align*}
                \mathbb{C}_{\text{eff}}
                =
                \mathbb{C}_{\text{m}}
                +
                c_{\text{i}}
                \left(
                    \mathbb{C}_{\text{i}}
                    - \mathbb{C}_{\text{m}}
                \right)
                \mathbb{A}_{\text{i}}^{\text{Approximated}}
            \end{align*}

        Parameters
        ----------
        c_i : float
            Volume fraction of inclusion.
        A_i : np.array (mandel6_4)
            Strain localization of inclusion.
        C_i : np.array (mandel6_4)
            Stiffness of inclusion.
        C_m : np.array (mandel6_4)
            Stiffness of matrix.

        Returns
        -------
        np.array (mandel6_4)
            Effective stiffness.

        """


#####################################################################################
class MoriTanaka(TwoPhaseComposite):
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
        r"""Wrap :meth:`Approximation.calc_A_SI_from_hill_polarization`"""
        return self.calc_A_SI_from_hill_polarization(
            P_i=self._P_i, C_i=self._C_i, C_m=self._C_m
        )


MoriTanaka.__doc__ = r"""
        Approximate strain localization following [Mori1973]_.

        The ansatz [Gross2016]_ (equation 8.101)

        .. math::
                \begin{align*}
                    \left<\mathbb{\varepsilon}\right>_{\text{i}}
                    =
                    \mathbb{A}^{\text{SI}}
                    \left[
                        \left< \mathbb{\varepsilon} \right>_{\text{m}}
                    \right]
                \end{align*}

        leads to

        .. math::
                \begin{align*}
                    \left<\mathbb{\varepsilon}\right>
                    &=
                    c_{\text{m}}
                    \left<\mathbb{\varepsilon}\right>_{\text{m}}
                    +
                    c_{\text{i}}
                    \left<\mathbb{\varepsilon}\right>_{\text{i}}    \\
                    &=
                    c_{\text{m}}
                    \left(\mathbb{A}^{\text{SI}}\right)^{-1}
                    \left<\mathbb{\varepsilon}\right>_{\text{i}}
                    +
                    c_{\text{i}}
                    \left<\mathbb{\varepsilon}\right>_{\text{i}}    \\
                    &=
                    \underbrace{
                        \left(
                            c_{\text{m}}
                            \left(\mathbb{A}^{\text{SI}}\right)^{-1}
                            +
                            c_{\text{i}}
                            \mathbb{I}^{\text{S}}
                        \right)
                    }_{\left( \mathbb{A}_{\text{i}}^{\text{MT}} \right)^{-1}}
                    \left<\mathbb{\varepsilon}\right>_{\text{i}}
                \end{align*}

        Use :math:`\mathbb{A}_{\text{i}}^{\text{Approximated}}`
        :math:`=\mathbb{A}_{\text{i}}^{\text{MT}}`
        .

        Note
        ----
            The Hill polarization of the inclusion
            :math:`\mathbb{P}_{\text{i}}`
            represents the geometry of the inclusion and depends on the
            material properties of the matrix.

        References
        ----------

        .. [Mori1973] Mori, T. and Tanaka, K., 1973. Average stress in matrix and
            average elastic energy of materials with misfitting inclusions.
            Acta metallurgica, 21(5), pp.571-574.

        .. [Gross2016] Gross, D. and Seelig, T., 2016. Lineare Bruchmechanik.
            In Bruchmechanik (pp. 69-162). Springer Vieweg, Berlin, Heidelberg.

        """

MoriTanaka.__init__.__doc__ = r"""
        Parameters
        ----------
        phases : dict
            Valid phases are 'inclusion' and 'matrix'.
        phases['inclusion']['material'] : material with .stiffness_mandel6
            Inclusion material with stiffness.
        phases['inclusion']['volume_fraction'] : float
            Volume fraction of inclusion.
        phases['inclusion']['hill_polarization'] : np.array (mandel6_4)
            Hill polarization of inclusion.
        phases['matrix']['material'] : material with .stiffness_mandel6
            Matrix material with stiffness.
        """

MoriTanaka.__init__.calc_A_MT_i = r"""
        Calc strain localization by Mori-Tanaka assumption.

       .. math::
           \begin{align*}
               \mathbb{A}_{\text{i}}^{\text{MT}}
               &=
               \left(
                   c_{\text{m}}
                   \left(
                       \mathbb{A}_{\text{i}}^{\text{SI}}
                   \right)^{-1}
                   +
                   c_{\text{i}}
                   \mathbb{I}^{\text{S}}
               \right)^{-1}
           \end{align*}

       Note
       ----
           See [Weng1990]_ (2.22) for a connection between
           average strain localization tensors of the Mori-Tanaka scheme
           and Hashin-Shtrikman-Walpole bounds, as the above equation can be
           cast into the following form:

           .. math::
               \begin{align*}
                   \mathbb{A}_{\text{i}}^{\text{MT}}
                   &=
                   \mathbb{A}_{\text{i}}^{\text{SI}}
                   \left(
                       c_{\text{m}}
                       \mathbb{I}^{\text{S}}
                       +
                       c_{\text{i}}
                       \mathbb{A}_{\text{i}}^{\text{SI}}
                   \right)^{-1}    \\
                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                   &=
                   \mathbb{A}_{\text{i}}^{\text{SI}}
                   \left(
                       \left<
                           \mathbb{A}^{\text{SI}}
                       \right>
                   \right)^{-1}  \;\;
                       \text{with
                       $\mathbb{A}_{\text{m}}^{\text{SI}}=\mathbb{I}$
                       }      \\
               \end{align*}

           Note that
           :math:`\mathbb{A}_{\text{i}}^{\text{SI}}
           (P, C_{\text{i}}, C_{\text{surrounding material}}
           )`
           is a function of the matrix stiffness in the Mori-Tanaka context
           and a function of the stiffness of the reference material in the
           Hashin-Shtrikman context.

       Parameters
       ----------
       c_i : float
           Volume fraction of inclusion.
       A_SI_i : np.array (mandel6_4)
           Strain localization of inclusion.

       Returns
       -------
       np.array (mandel6_4)
           Strain localization Mori-Tanaka.

       References
       ----------

       .. [Weng1990] Weng, G. J. (1990).
           The theoretical connection between Mori-Tanaka's theory and the
           Hashin-Shtrikman-Walpole bounds.
           International Journal of Engineering Science, 28(11), 1111-1120.

       """

MoriTanaka.calc_C_eff.__doc__ = r"""
        Calc, set as attribute and return: Effective stiffness

        Returns
        -------
        np.array (mandel6_4)
            Effective stiffness
        """


class MoriTanakaOrientationAveraged(MoriTanaka):
    r"""Include fiber orientation in Mori-Tanaka approximation
    following [Brylka2017]_ equation (2.73).

    Use :math:`\mathbb{A}_{\text{i}}^{\text{Approximated}}`
    :math:`=\left<\mathbb{A}_{\text{i}}^{\text{MT}}\right>_{\text{f}}`
    with

    .. math::
            \begin{align*}
                \left<\mathbb{A}_{\text{i}}^{\text{MT}}\right>_{\text{f}}
                =
                \left(
                    c_{\text{m}}
                    \left(
                        \left<
                            \mathbb{A}_{\text{i}}^{\text{SI}}
                        \right>_{\text{f}}
                    \right)^{-1}
                    +
                    c_{\text{i}}
                    \mathbb{I}^{\text{S}}
                \right)^{-1}
            \end{align*}

    and with the orientation averaged strain localization of the
    single inclusion problem
    :math:`\left<\mathbb{A}_{\text{i}}^{\text{SI}}\right>_{\text{f}}`.

    References
    ----------

    .. [Brylka2017] Brylka, B., Charakterisierung und Modellierung der
        Steifigkeit von langfaserverstärktem Polypropylen. 10, (2017).

    Examples
    --------
    >>> import mechkit
    >>> inp = {
            'E_f': 73.0,
            'E_m': 3.4,
            'N4': np.array(
              [[ 4.09e-01,  1.48e-01,  1.03e-02, -2.20e-03, -1.86e-02,  3.52e-02],
               [ 1.48e-01,  2.51e-01,  6.50e-03, -2.00e-03, -5.50e-03,  3.11e-02],
               [ 1.03e-02,  6.50e-03,  9.70e-03,  8.00e-04, -1.20e-03,  4.00e-04],
               [-2.20e-03, -2.00e-03,  8.00e-04,  1.30e-02,  5.00e-04, -7.70e-03],
               [-1.86e-02, -5.50e-03, -1.20e-03,  5.00e-04,  2.06e-02, -3.20e-03],
               [ 3.52e-02,  3.11e-02,  4.00e-04, -7.70e-03, -3.20e-03,  2.97e-01]]
              ),
            'c_f': 0.22,
            'nu_f': 0.22,
            'nu_m': 0.385,
            }

    >>> inclusion = mechkit.material.Isotropic(
                E=inp['E_f'],
                nu=inp['nu_f'],
                )
    >>> matrix = mechkit.material.Isotropic(
                E=inp['E_m'],
                nu=inp['nu_m'],
                )
    >>> averager = mechmean.orientation_averager.AdvaniTucker(N4=inp['N4'])
    >>> P_func = mechmean.hill_polarization.Castaneda().needle

    >>> input_dict = {
            'phases': {
                'inclusion': {
                    'material': inclusion,
                    'volume_fraction': inp['c_f'],
                    'hill_polarization': P_func(matrix=matrix),
                    },
                'matrix': {
                    'material': matrix,
                    },
                },
            'averaging_func': averager.average,
            }

    >>> mori = mechmean.approximation.MoriTanakaOrientationAveraged(**input_dict)
    >>> C_eff = mori.calc_C_eff()
    [[ 1.479e+01,  7.000e+00,  5.045e+00, -3.118e-02, -2.842e-01,  4.863e-01],
     [ 7.000e+00,  1.251e+01,  5.057e+00, -2.973e-02, -7.226e-02,  4.390e-01],
     [ 5.045e+00,  5.057e+00,  8.827e+00,  1.472e-02, -1.303e-02, -2.303e-02],
     [-3.118e-02, -2.973e-02,  1.472e-02,  3.824e+00,  1.668e-02, -1.239e-01],
     [-2.842e-01, -7.226e-02, -1.303e-02,  1.668e-02,  3.982e+00, -4.843e-02],
     [ 4.863e-01,  4.390e-01, -2.303e-02, -1.239e-01, -4.843e-02,  8.386e+00]])

    """

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        ....
            See super class and in addition:
        averaging_func : function
            Function averaging fourth order tensor.
        """
        self.averaging_func = kwargs["averaging_func"]
        super().__init__(**kwargs)
        self._dC = self._C_i - self._C_m

    def calc_A_SI_i(self):
        r"""
        Some orientation averaging schemes are restricted to quantities
        with transversal isotropic material symmetry.
        :math:`\mathbb{A}_{\text{i}}^{\text{SI}}`
        may lack this symmetry.
        The orientation averaging scheme is applied to an
        intermediate, stiffness-like quantity
        :math:`\mathbb{A}^{\star}`
        leading to

       .. math::
               \begin{align*}
                   \mathbb{A}^{\star}
                   &=
                   \text{d}\mathbb{C}
                   \mathbb{A}^{\text{SI}}           \\
                   \left<\mathbb{A}^{\star}\right>_{\text{f}}
                   &=
                   \text{Orientation average}\left(\mathbb{A}^{\star}\right) \\
                   \left<\mathbb{A}_{\text{i}}^{\text{SI}}\right>_{\text{f}}
                   &=
                   \left(\text{d}\mathbb{C}\right)^{-1}
                   \left<\mathbb{A}^{\star}\right>_{\text{f}}   \\
                   \text{with } \;\;
                   \text{d}\mathbb{C}
                   &=
                   \mathbb{C}_{\text{i}} - \mathbb{C}_{\text{m}}.
               \end{align*}

        """
        A_SI = self.calc_A_SI_from_hill_polarization(
            P_i=self._P_i, C_i=self._C_i, C_m=self._C_m
        )
        A_star = self._dC @ A_SI
        A_star_averaged = self.averaging_func(A_star)
        return inv(self._dC) @ A_star_averaged


class Kehrer2019(object):
    r"""Two-step Hashin-Shtrikman homogenization scheme for fiber reinforced
    composites [Kehrer2019]_

    **Step 1**

    .. math::
            \begin{align*}
                \mathbb{C}_{\text{upp}}
                &=
                \mathbb{C}_{\text{i}}
                +
                c_{\text{m}}
                \text{d} \mathbb{C}
                \left(
                    \mathbb{I}^{\text{S}}
                    -
                    c_{\text{i}}
                    \mathbb{P}_{\text{i}}^{\text{UD}}
                    \text{d} \mathbb{C}
                \right)^{-1}        \\
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                \mathbb{C}_{\text{low}}
                &=
                \mathbb{C}_{\text{m}}
                -
                c_{\text{i}}
                \text{d} \mathbb{C}
                \left(
                    \mathbb{I}^{\text{S}}
                    -
                    c_{\text{m}}
                    \mathbb{P}_{\text{m}}^{\text{UD}}
                    \text{d} \mathbb{C}
                \right)^{-1}        \\
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                \text{d}\mathbb{C}
                &=
                \mathbb{C}_{\text{m}}-\mathbb{C}_{\text{i}}
            \end{align*}

    with :math:`\text{j} \in [\text{i},\text{m}]` and

        - :math:`c_{\text{j}}` : Volume fraction of phase j
        - :math:`\mathbb{C}_{\text{j}}` : Stiffness of phase j
        - :math:`\mathbb{P}_{\text{j}}^{\text{UD}}` : Hill polarization of needle shape in material of phase j

    **Step 2**

    For :math:`\text{j} \in [\text{upp},\text{low}]`

    .. math::
            \begin{align*}
                \mathbb{C}_{\text{j}}^{\text{eff}}
                &=
                \mathbb{C}_{\text{0}}
                -
                \mathbb{P}_{\text{0}}^{-1}
                +
                \left<
                    \mathbb{A}_{\text{j}}^{\star}
                \right>_{\text{orientation}}^{-1}       \\
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                \mathbb{A}_{\text{j}}^{\star}
                &=
                \left(
                    \mathbb{P}_{\text{0}}^{-1}
                    +
                    \mathbb{C}_{\text{j}}
                    -
                    \mathbb{C}_{\text{0}}
                \right)^{-1}                            \\
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                \mathbb{C}_{\text{0}}
                &=
                \left(
                    1-k
                \right)
                \mathbb{C}_{\text{m}}
                +
                k
                \mathbb{C}_{\text{i}}
            \end{align*}

    with

        - :math:`k` :  Scalar weight of fiber phase in reference material.
        - :math:`\mathbb{C}_{\text{upp}}^{\text{eff}}` : Upper effective stiffness
        - :math:`\mathbb{C}_{\text{low}}^{\text{eff}}` : Lower effective stiffness

    References
    ----------

    .. [Kehrer2019] Kehrer, M.L., 2019. Thermomechanical Mean-Field
        Modeling and Experimental Characterization of Long Fiber-Reinforced
        Sheet Molding Compound Composites (Vol. 15).
        KIT Scientific Publishing.

    Examples
    --------
    >>> import mechkit
    >>> inp = {
            'E_f': 73.0,
            'E_m': 3.4,
            'N4': np.array(
              [[ 4.09e-01,  1.48e-01,  1.03e-02, -2.20e-03, -1.86e-02,  3.52e-02],
               [ 1.48e-01,  2.51e-01,  6.50e-03, -2.00e-03, -5.50e-03,  3.11e-02],
               [ 1.03e-02,  6.50e-03,  9.70e-03,  8.00e-04, -1.20e-03,  4.00e-04],
               [-2.20e-03, -2.00e-03,  8.00e-04,  1.30e-02,  5.00e-04, -7.70e-03],
               [-1.86e-02, -5.50e-03, -1.20e-03,  5.00e-04,  2.06e-02, -3.20e-03],
               [ 3.52e-02,  3.11e-02,  4.00e-04, -7.70e-03, -3.20e-03,  2.97e-01]]
              ),
            'c_f': 0.22,
            'k': 0.5,
            'nu_f': 0.22,
            'nu_m': 0.385,
            }

    >>> inclusion = mechkit.material.Isotropic(
                E=inp['E_f'],
                nu=inp['nu_f'],
                )
    >>> matrix = mechkit.material.Isotropic(
                E=inp['E_m'],
                nu=inp['nu_m'],
                )
    >>> averager = mechmean.orientation_averager.AdvaniTucker(N4=inp['N4'])

    >>> input_dict = {
            'phases': {
                'inclusion': {
                    'material': inclusion,
                    'volume_fraction': inp['c_f'],
                    },
                'matrix': {
                    'material': matrix,
                    'volume_fraction': 1. - inp['c_f'],
                    },
                },
            'k': inp['k'],
            'averaging_func': averager.average,
            }
    >>> hashin = mechmean.approximation.Kehrer2019(**input_dict)
    >>> C_eff = hashin.calc_C_eff()
    Effective_stiffness(
    upper=array(
    [[ 1.793e+01,  6.792e+00,  6.440e+00, -7.925e-03, -1.225e-01,  2.797e-01],
     [ 6.792e+00,  1.666e+01,  6.502e+00, -1.443e-02, -9.110e-03,  2.566e-01],
     [ 6.440e+00,  6.502e+00,  1.438e+01, -2.853e-03, -4.822e-02, -3.038e-02],
     [-7.925e-03, -1.443e-02, -2.853e-03,  8.566e+00,  1.067e-01, -7.336e-02],
     [-1.225e-01, -9.110e-03, -4.822e-02,  1.067e-01,  8.955e+00, -1.951e-02],
     [ 2.797e-01,  2.566e-01, -3.038e-02, -7.336e-02, -1.951e-02,  1.099e+01]]
    ),
    lower=array(
    [[ 1.320e+01,  6.537e+00,  5.063e+00, -2.566e-02, -2.110e-01,  4.354e-01],
     [ 6.537e+00,  1.134e+01,  5.077e+00, -2.310e-02, -5.618e-02,  3.731e-01],
     [ 5.063e+00,  5.077e+00,  8.767e+00,  8.965e-03, -8.823e-03, -1.952e-02],
     [-2.566e-02, -2.310e-02,  8.965e-03,  3.737e+00,  1.406e-02, -9.265e-02],
     [-2.110e-01, -5.618e-02, -8.823e-03,  1.406e-02,  3.847e+00, -3.933e-02],
     [ 4.354e-01,  3.731e-01, -1.952e-02, -9.265e-02, -3.933e-02,  7.122e+00]]
    )   )

    """

    def __init__(self, phases, averaging_func, k):
        r"""
        Parameters
        ----------
        phases : dict
            Valid phases are 'inclusion' and 'matrix'.
        phases['inclusion']['material'] : mechkit.material.Isotropic
            Isotropic inclusion material.
        phases['inclusion']['volume_fraction'] : float
            Volume fraction of inclusion.
        phases['matrix']['material'] : mechkit.material.Isotropic
            Isotropic matrix material.
        averaging_func : callable
            Callable averaging fourth order tensor.
        k : float in range(0,1)
            Weight of fiber phase in reference material.
        """

        self.phases = phases
        self.averaging_func = averaging_func
        self.k = k

        self.Pud_func = mechmean.hill_polarization.Castaneda().needle

        self.P0_func = functools.partial(
            mechmean.hill_polarization.Castaneda().spheroid, aspect_ratio=1.0
        )

        self._I4s = mechkit.notation.Converter().to_mandel6(mechkit.tensors.Basic().I4s)

    def calc_C_eff(self):
        r"""Calc, set as attribute and return: Effective stiffnesses

        Returns
        -------
        tuple(np.array, np.array) with arrays of shape (mandel6_4)
            Upper and lower effective stiffness
        """
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
    r"""Hashin-Shtrikman scheme formulated by
    [Walpole1966_I]_, [Walpole1966_II]_
    formatted following
    [Fernandez2019]_ (equation 22)
    and applicable to non-spherical Hill polarization following
    [Walpole1969]_, [Willis1977]_ (page 190) and [Willis1981]_ (pages 36, 38)
    including a dual scheme for singular polarizations following
    [Walpole1969]_ (page 238).

    .. math::
            \begin{align*}
                \mathbb{C}^{\text{eff}}
                &=
                \mathbb{C}_{\text{0}}
                -
                \mathbb{P}_{\text{0}}^{-1}
                +
                \left<
                    \mathbb{W}
                \right>^{-1}        \\
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                \mathbb{W}
                &=
                \left(
                    \mathbb{C}
                    -
                    \mathbb{C}_{\text{0}}
                    +
                    \mathbb{P}_{\text{0}}^{-1}
                \right)^{-1}
            \end{align*}

    with

        - :math:`\mathbb{C}_{\text{0}}` : Stiffness of reference material
        - :math:`\mathbb{P}_{\text{0}}` : Hill polarization of single inclusion in reference material
        - :math:`\mathbb{C}` : Stiffness at location :math:`\boldsymbol{x}` (piecewise constant phases)
        - :math:`\left<\right>` : Suitable average for given microstructure. See e.g. [Walpole1966_II] equations 24, 25

    Note
    ----

        The Hill polarization
        :math:`\mathbb{P}_{\text{0}}`
        reflects the symmetry of the two-point statistics
        (See [Willis1977]_ (page 188), [Fernandez2019]_ ).

    References
    ----------

    .. [Walpole1966_I] Walpole, L. J. (1966). On bounds for the overall
        elastic moduli of inhomogeneous systems—I.
        Journal of the Mechanics and Physics of Solids, 14(3), 151-162.

    .. [Walpole1966_II] Walpole, L. J. (1966). On bounds for the overall
        elastic moduli of inhomogeneous systems—II.
        Journal of the Mechanics and Physics of Solids, 14(5), 289-301.

    .. [Fernandez2019] Fernández, M. L., & Böhlke, T. (2019).
        Representation of Hashin–Shtrikman Bounds in Terms of Texture
        Coefficients for Arbitrarily Anisotropic Polycrystalline Materials.
        Journal of Elasticity, 134(1), 1-38.

    .. [Walpole1969] Walpole, L. J. (1969). On the overall
        elastic moduli of composite materials.
        Journal of the Mechanics and Physics of Solids, 17(4), 235-251.

    .. [Willis1977] Willis, J. R. (1977). Bounds and self-consistent
        estimates for the overall properties of anisotropic composites.
        Journal of the Mechanics and Physics of Solids, 25(3), 185-202.

    .. [Willis1981] Willis, J. R. (1981). Variational and related methods
        for the overall properties of composites.
        In Advances in applied mechanics (Vol. 21, pp. 1-78). Elsevier.

    """

    def __init__(self, phases, P_func, averaging_func):
        r"""
        Parameters
        ----------
        phases : nested dict
            Each entry represents one phase.
        phases[....]['material'] : material with .stiffness_mandel6
            Material of phase '...'.
        phases[....]['W'] : None
            Phases must not have key 'W', because each phase-dict is
            extended by 'W'.
            The resulting extended dict is passed to averging_func.
        P_func : callcable
            Callable returning Hill polarization for given material.
        averaging_func : callable
            Callable that takes nested dict of phases and calculates
            average of 'W'.
        """
        self.P_func = P_func
        for key, phase in phases.items():
            phases[key]["stiffness"] = phase["material"].stiffness_mandel6
        self.phases = phases
        self.averaging_func = averaging_func

    def calc_C_eff(self, ref_material):
        r"""Calc effective stiffness for given reference material

        Handle singular polarizations.

        Returns
        -------
        np.array (mandel6_4)
            Effective stiffness
        """
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
        r"""
        Calc **strain localization tensor**

        .. math::
            \begin{align*}
                \mathbb{A}_{\text{j}}
                =
                \mathbb{A}_{\text{j}}^{\text{SI}}
                \left<
                \mathbb{A}^{\text{SI}}
                \right>^{-1}
            \end{align*}

        mapping average strain to average strain in phase j.

        .. math::
            \begin{align*}
                \left<\mathbb{\varepsilon}\right>_{\text{j}}
                =
                \mathbb{A}_{\text{j}}
                \left[
                \left<\mathbb{\varepsilon}\right>
                \right]
            \end{align*}

        with

            - :math:`\mathbb{A}^{\text{SI}}` : Strain localization tensor in single inclusion problem
            - :math:`\left<\right>` : Suitable average for given microstructure
            - :math:`\left<\mathbb{\varepsilon}\right>_{\text{j}}` : Average strain in phase j
            - :math:`\left<\mathbb{\varepsilon}\right>` : Average strain

        Parameters
        ----------
        key_phase : str
            Key of phase j in dict *phases*
        ref_material : mechkit.material.Isotropic
            Reference material

        Returns
        -------
        np.array (mandel6_4)
            Strain localization
        """
        P = self._calc_P(ref_material=ref_material)
        C_0 = ref_material.stiffness_mandel6
        phases = copy.deepcopy(self.phases)

        func_A_SI = TwoPhaseComposite().calc_A_SI_from_hill_polarization

        for key, phase in phases.items():
            phases[key]["A_SI"] = func_A_SI(P_i=P, C_i=phase["stiffness"], C_m=C_0)

        A_SI_average = self.averaging_func(phases=phases, key_to_be_averaged="A_SI")

        return phases[key_phase]["A_SI"] @ inv(A_SI_average)

    def calc_stress_localization_tensor(self, key_phase, ref_material):
        r"""
        Calc **stress localization tensor**

        .. math::
            \begin{align*}
                \mathbb{B}_{\text{j}}
                =
                \mathbb{C}_{\text{j}}
                \mathbb{A}_{\text{j}}
                \mathbb{C}_{\text{0}}^{-1}
            \end{align*}

        mapping average stress to average stress in phase j.

        .. math::
            \begin{align*}
                \left<\mathbb{\sigma}\right>_{\text{j}}
                =
                \mathbb{B}_{\text{j}}
                \left[
                \left<\mathbb{\sigma}\right>
                \right]
            \end{align*}

        with

            - :math:`\mathbb{C}_{\text{j}}` : Stiffness of phase j
            - :math:`\mathbb{A}_{\text{j}}` : Strain localization tensor of phase j
            - :math:`\mathbb{C}_{\text{0}}` :  Stiffness of reference material
            - :math:`\left<\mathbb{\sigma}\right>_{\text{j}}` : Average stress in phase j
            - :math:`\left<\mathbb{\sigma}\right>` : Average stress

        Parameters
        ----------
        key_phase : str
            Key of phase j in dict *phases*
        ref_material : mechkit.material.Isotropic
            Reference material

        Returns
        -------
        np.array (mandel6_4)
            Stress localization
        """
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
    r""":class:`.HashinShtrikmanWalpole` with volume average is weightet summation"""

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
    r""":class:`.HashinShtrikmanWalpole` with orientation average over single phase"""

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
        r"""
        Parameters
        ----------
        ....
            See super class and in addition:
        orientation_averager : callable
            Returns orientation average of fourth order tensor given as first
            positional argument
        """
        super().__init__(
            phases=phases,
            P_func=P_func,
            averaging_func=self.AveragerOrientationWithKeyword(
                averager=orientation_averager,
            ),
        )


class Kehrer2019HSW(object):
    r"""Implementation of Kehrer2019 using class HashinShtrikmanWalpole"""

    def __init__(self, phases, averaging_func, k):
        r"""See :class:`.Kehrer2019`"""
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
        r"""See :class:`.Kehrer2019`"""
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
    r"""Based on Kehrer2019 but with generic structure"""

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


if __name__ == "__main__":
    np.set_printoptions(linewidth=140)

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
        "k": 1.0,
        "nu_f": 0.22,
        "nu_m": 0.385,
    }

    ###########################
    # Format input

    inclusion = mechkit.material.Isotropic(E=inp["E_f"], nu=inp["nu_f"])

    matrix = mechkit.material.Isotropic(E=inp["E_m"], nu=inp["nu_m"])

    ###########################
    # Start

    input_dict = {
        "phases": {
            "i": {
                "stiffness": inclusion.stiffness_mandel6,
                "volume_fraction": inp["c_f"],
            },
            "m": {
                "stiffness": matrix.stiffness_mandel6,
                "volume_fraction": 1.0 - inp["c_f"],
            },
        },
        "P_func": functools.partial(
            mechmean.hill_polarization.Castaneda().spheroid, aspect_ratio=1.0
        ),
    }

    ###########################

    hashin = mechmean.approximation.HSW_VolumeFraction(**input_dict)

    C_eff = hashin.calc_C_eff(ref_material=matrix)

    A = {
        key: hashin.calc_strain_localization_tensor(key_phase=key, ref_material=matrix)
        for key in ["i", "m"]
    }

    A_av = np.zeros((6, 6))
    for key, val in input_dict["phases"].items():
        A_av += val["volume_fraction"] * A[key]

    printQueue = [
        "A",
        "A_av",
    ]

    # Print
    for val in printQueue:
        print(val)
        pprint.pprint(eval(val))
        print()
