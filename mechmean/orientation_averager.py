#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orientation averager
"""

import numpy as np
import mechkit
import mechmean


class AdvaniTucker:
    r"""Orientation averaging following [Advani1987]_

    The orientation average of a fourth order tensor
    :math:`\left<\mathbb{A}\right>_{\text{orientation}}`
    is calculated based on fabric tensors of the first kind
    of order two
    :math:`\boldsymbol{N}`
    and four
    :math:`\mathbb{N}`.

    If the symmetry axis of :math:`\mathbb{A}` is aligned with the base
    direction of the first tensor index
    :math:`\boldsymbol{e}_{0}`,
    the average is given by

        .. math::
            \begin{align}
                \left<\mathbb{A}\right>_{\text{orientation}}
                &=
                b_0\mathbb{N}             \\
                &+
                b_1 \left(
                        \boldsymbol{N} \otimes \boldsymbol{I}
                    +   \boldsymbol{I} \otimes \boldsymbol{N}
                    \right)         \\
                &+
                b_2 \left(
                        \boldsymbol{N} \Box \boldsymbol{I}
                    +   \left( \boldsymbol{N} \Box \boldsymbol{I} \right)^{T_\text{R}}
                    +   \left( \boldsymbol{I} \Box \boldsymbol{N} \right)^{T_\text{H}}
                    +   \left( \boldsymbol{I} \Box \boldsymbol{N} \right)^{T_\text{R}}
                    \right)         \\
                &+
                b_3 \boldsymbol{I} \otimes \boldsymbol{I} \\
                &+
                b_4 \mathbb{I}^{\text{S}}
            \end{align}

    with coefficients

        .. math::
            \begin{align}
                b_0     &= A_{0000} + A_{1111} - 2A_{0011} - 4A_{0101}                  \\
                b_1     &= A_{0011} - A_{1122}                                          \\
                b_2     &= A_{0101} + \frac{1}{2} \left( A_{1122} - A_{1111} \right)    \\
                b_3     &= A_{1122}                                                     \\
                b_4     &= A_{1111} - A_{1122}
            \end{align}

    References
    ----------

    .. [Advani1987] Advani, S.G. and Tucker III, C.L., 1987.
        The use of tensors to describe and predict fiber orientation in short
        fiber composites. Journal of rheology, 31(8), pp.751-784.

    """

    def __init__(self, N4):
        self._con = mechkit.notation.Converter()
        self.base = self.get_base(N4)

    def get_base(self, N4):
        """Calc bases of factors :math:`b_i`"""
        tensors = mechkit.tensors.Basic()
        I2 = tensors.I2
        I4s = tensors.I4s

        N4 = self._con.to_tensor(N4)
        N2 = np.einsum("ijkl, kl->ij", N4, I2)

        def dyad(A, B):
            return np.einsum("ij, kl->ijkl", A, B)

        def box(A, B):
            return np.einsum("ij, kl->iklj", A, B)

        N2_box_I2 = box(N2, I2)
        I2_box_N2 = box(I2, N2)

        base = np.zeros((5, 3, 3, 3, 3))
        base[0, :] = N4

        base[1, :] = dyad(N2, I2) + dyad(I2, N2)

        base[2, :] = (
            N2_box_I2
            + np.einsum("ijkl->ijlk", N2_box_I2)
            + np.einsum("ijkl->klij", I2_box_N2)
            + np.einsum("ijkl->ijlk", I2_box_N2)
        )
        base[3, :] = dyad(I2, I2)

        base[4, :] = I4s

        return base

    def average(self, B):
        """Calc average of B"""

        # TODO: Average both B4 and B2
        # TODO: Ask for N-tensor of same order as B-tensor
        #       and calc lower orders of N by contraction

        A = self._con.to_tensor(B)

        b = np.zeros((5))
        b[0] = A[0, 0, 0, 0] + A[1, 1, 1, 1] - 2.0 * A[0, 0, 1, 1] - 4.0 * A[0, 1, 0, 1]
        b[1] = A[0, 0, 1, 1] - A[1, 1, 2, 2]
        b[2] = A[0, 1, 0, 1] + 0.5 * (A[1, 1, 2, 2] - A[1, 1, 1, 1])
        b[3] = A[1, 1, 2, 2]
        b[4] = A[1, 1, 1, 1] - A[1, 1, 2, 2]

        av = np.einsum("m, mijkl->ijkl", b, self.base)

        return self._con.to_mandel6(av)
