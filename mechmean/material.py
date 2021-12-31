#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Material related operations
"""

import numpy as np
import mechkit

tensors = mechkit.tensors.Basic()


class SpecificMaterial(object):
    def __init__(self):
        self.con = mechkit.notation.VoigtConverter(silent=True)

    def _copy_upper_triangle(self, matrix):
        r"""Copy upper triangle to lower triangle, i.e. make symmetric"""
        index_lower_triangle = np.tril_indices(6, -1)
        matrix[index_lower_triangle] = matrix.T[index_lower_triangle]
        return matrix

    def _voigt_to_tensor(self, stiffness):
        return self.con.to_tensor(
            self.con.voigt_to_mandel6(stiffness, voigt_type="stiffness")
        )


class AlignedStiffnessFactory(SpecificMaterial):
    def __init__(self):
        self.number_independent_param = {"hexagonal_axis1": 5}
        self.func = {"hexagonal_axis1": self.hexagonal_axis1}
        super().__init__()

    def positiv_definit(self, label):
        nbr_param = self.number_independent_param[label]
        func = self.func[label]

        eigen = np.array([-1])
        while not all(eigen > 0):
            C = func(*np.random.rand(nbr_param).tolist())
            eigen = np.linalg.eig(C)[0]
        return C

    def hexagonal_axis1(self, C1111, C1122, C1133, C2222, C1313):
        voigt_half = np.array(
            [
                [C1111, C1122, C1122, 0, 0, 0],
                [0, C2222, C1133, 0, 0, 0],
                [0, 0, C2222, 0, 0, 0],
                [0, 0, 0, C2222 - C1133, 0, 0],
                [0, 0, 0, 0, C1313, 0],
                [0, 0, 0, 0, 0, C1313],
            ],
            dtype="float64",
        )
        voigt = self._copy_upper_triangle(matrix=voigt_half)
        return self.con.voigt_to_mandel6(voigt, voigt_type="stiffness")
