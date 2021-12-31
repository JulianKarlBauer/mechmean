#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orientation averager
"""

import numpy as np
import mechkit
import mechmean


class AdvaniTucker_in_kanatani_third_kind(mechmean.orientation_averager.AdvaniTucker):
    def __init__(self, D2, D4):
        self._con = mechkit.notation.Converter()
        self.base = self.get_base(D2, D4)

    def get_base(self, D2, D4):

        tensors = mechkit.tensors.Basic()
        I2 = tensors.I2
        I4s = tensors.I4s
        P1 = tensors.P1
        sym = mechmean.operators.sym

        D4 = self._con.to_tensor(D4)
        D2 = self._con.to_tensor(D2)

        def dyad(A, B):
            return np.einsum("ij, kl->ijkl", A, B)

        def box(A, B):
            return np.einsum("ij, kl->iklj", A, B)

        D_box_I = box(D2, I2)
        I_box_D = box(I2, D2)

        D_dyad_I = dyad(D2, I2)
        I_dyad_D = dyad(I2, D2)

        base = np.zeros((5, 3, 3, 3, 3))
        base[0, :] = 8.0 / 315.0 * D4 + 4.0 / 35.0 * sym(I_dyad_D) + 3.0 / 5.0 * sym(P1)

        base[1, :] = 2.0 / 15.0 * (D_dyad_I + I_dyad_D) + 2.0 * P1

        base[2, :] = (
            2.0
            / 15.0
            * (
                D_box_I
                + np.einsum("ijkl->ijlk", D_box_I)
                + np.einsum("ijkl->klij", I_box_D)
                + np.einsum("ijkl->ijlk", I_box_D)
            )
            + 4.0 / 3.0 * I4s
        )

        base[3, :] = dyad(I2, I2)

        base[4, :] = I4s

        return base
