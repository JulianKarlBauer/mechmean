#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Docstrings for classes and methods in approximation.py
"""
import mechmean
import mechmean.approximation as approx

docstrings = {
    approx.TwoPhaseComposite: r"""
        Base class for approximations of two-phase materials.

        Formulations are based on average **strain localization tensors**
        :math:\mathbb{A}_{\text{j}}`
        mapping the average strain to the average strain in phase j,
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

        """,
    approx.TwoPhaseComposite.calc_A_SI_from_hill_polarization: r"""
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
            Hill polarization
        C_i : np.array (mandel6_4)
            Stiffness of inclusion
        C_m : np.array (mandel6_4)
            Stiffness of matrix

        Returns
        -------
        np.array (mandel6_4)
            Strain localization in single inclusion problem.

        """,
    approx.TwoPhaseComposite.calc_C_eff_by_A_i: r"""
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

        """,
    #############################################################
    approx.MoriTanaka: r"""
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

        .

        Note
        ----
            The Hill polarization of the inclusion
            :math:`\mathbb{P}_{\text{i}}`
            represents the geometry of the inclusion and depends on the
            material properties of the matrix.

        """,
    approx.MoriTanaka.__init__: r"""
        Parameters
        ----------
        phases : dict
            Valid phases are 'inclusion' and 'matrix'.
        phases['inclusion']['material'] : mechkit.material.Isotropic
            Inclusion material
        phases['inclusion']['volume_fraction'] : float
            Volume fraction of inclusion
        phases['inclusion']['hill_polarization'] : np.array (mandel6_4)
            Hill polarization
        phases['matrix']['material'] : mechkit.material.Isotropic
            Matrix material
        """,
    approx.MoriTanaka.calc_A_MT_i: r"""
        Calc strain localization by Mori-Tanaka assumption with

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
            and Hashin-Shtrikman-Walpole scheme, as the above equation can be
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

        """,
    approx.MoriTanaka.calc_C_eff: r"""
        Calc, set as attribute and return: Effective stiffness

        Returns
        -------
        np.array (mandel6_4)
            Effective stiffness

        """,
    #############################################################
    approx.MoriTanakaOrientationAveragedBenveniste: r"""
        Include fiber orientation in Mori-Tanaka approximation
        following [Benveniste1987]_ and [Brylka2017]_ equation (2.73).

        Use
        :math:`\left<\mathbb{A}_{\text{i}}^{\text{MT}}\right>_{\text{f}}`
        as
        :math:`\mathbb{A}_{\text{i}}^{\text{Approximated}}`
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

        >>> mori = mechmean.approximation.MoriTanakaOrientationAveragedBenveniste(**input_dict)
        >>> C_eff = mori.calc_C_eff()
        [[ 1.479e+01,  7.000e+00,  5.045e+00, -3.118e-02, -2.842e-01,  4.863e-01],
         [ 7.000e+00,  1.251e+01,  5.057e+00, -2.973e-02, -7.226e-02,  4.390e-01],
         [ 5.045e+00,  5.057e+00,  8.827e+00,  1.472e-02, -1.303e-02, -2.303e-02],
         [-3.118e-02, -2.973e-02,  1.472e-02,  3.824e+00,  1.668e-02, -1.239e-01],
         [-2.842e-01, -7.226e-02, -1.303e-02,  1.668e-02,  3.982e+00, -4.843e-02],
         [ 4.863e-01,  4.390e-01, -2.303e-02, -1.239e-01, -4.843e-02,  8.386e+00]])

        """,
    approx.MoriTanakaOrientationAveragedBenveniste.__init__: r"""
        Parameters
        ----------
        ....
            Parameters of
            :any:`mechmean.approximation.MoriTanaka`
            and in addition:
        averaging_func : function
            Function averaging fourth order tensor.

        """,
    approx.MoriTanakaOrientationAveragedBenveniste.calc_A_SI_i: r"""
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

        """,
    #############################################################
    approx.Kehrer2019: r"""
        Two-step Hashin-Shtrikman homogenization scheme for fiber reinforced
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

        """,
    approx.Kehrer2019.__init__: r"""
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
        """,
    approx.Kehrer2019.calc_C_eff: r"""
        Calc, set as attribute and return: Effective stiffnesses

        Returns
        -------
        tuple(np.array, np.array) with arrays of shape (mandel6_4)
            Upper and lower effective stiffness
        """,
    #############################################################
    approx.HashinShtrikmanWalpole: r"""
        Hashin-Shtrikman scheme formulated by
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

        """,
    approx.HashinShtrikmanWalpole.__init__: r"""
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
        """,
    approx.HashinShtrikmanWalpole.calc_C_eff: r"""
        Calc effective stiffness for given reference material

        Handle singular polarizations.

        Returns
        -------
        np.array (mandel6_4)
            Effective stiffness
        """,
    approx.HashinShtrikmanWalpole.calc_strain_localization_tensor: r"""
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
        """,
    approx.HashinShtrikmanWalpole.calc_stress_localization_tensor: r"""
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
        """,
    #############################################################
    approx.HSW_VolumeFraction: r"""
        :any:`mechmean.approximation.HashinShtrikmanWalpole`
        with volume average calculated as weighted summation""",
    #############################################################
    approx.HSW_SinglePhaseOrientation: r"""
        :any:`mechmean.approximation.HashinShtrikmanWalpole`
        with orientation average over single phase""",
    approx.HSW_SinglePhaseOrientation.__init__: r"""
        Parameters
        ----------
        ....
            Parameters of
            :any:`mechmean.approximation.HashinShtrikmanWalpole`
            and in addition:
        orientation_averager : callable
            Returns orientation average of fourth order tensor given as first
            positional argument
        """,
    #############################################################
    approx.Kehrer2019HSW: r"""
        Implementation of :any:`mechmean.approximation.Kehrer2019` using
        :any:`mechmean.approximation.HashinShtrikmanWalpole`
        """,
    approx.Kehrer2019HSW.__init__: r"""
        See
        :any:`mechmean.approximation.Kehrer2019`
        """,
    approx.Kehrer2019HSW.calc_C_eff: r"""
        See
        :any:`mechmean.approximation.Kehrer2019`
        """,
    #############################################################
    approx.HSW2StepInterpolatedReferenceMaterial: r"""
        Based on Kehrer2019 but with interpolation parameter in both steps
        combined with the inverse scheme for singular polarizations
        leading to a generic structure
        """,
}

for identifier, docstring in docstrings.items():
    setattr(identifier, "__doc__", docstring)
