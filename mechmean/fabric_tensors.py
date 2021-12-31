#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mechkit
import mechmean


class KanataniFactory(object):
    def __init__(self, N):
        self.con = mechkit.notation.Converter()
        self._I2 = mechkit.tensors.Basic().I2

        self.N = N = self.con.to_tensor(N)
        self.degree = len(N.shape)

        degrees = [x for x in range(1, self.degree + 1) if x % 2 == 0]
        for degree in reversed(degrees):
            N = self.first_kind(degree)
            setattr(self, "N{}".format(degree), N)
            setattr(self, "F{}".format(degree), self.second_kind(N))
            setattr(self, "D{}".format(degree), self.third_kind(N))

    def __getitem__(self, key):
        """Make attributes accessible dict-like."""
        return getattr(self, key)

    def first_kind(self, degree):
        nbr_times_decrease = int((self.degree - degree) / 2)
        N = self.N
        for i in range(nbr_times_decrease):
            N = self.decrease_first_kind_by_one_degree(N)
        return N

    def decrease_first_kind_by_one_degree(self, N):
        return np.einsum("...ij, ...ij->...", N, self._I2)

    def second_kind(self, N):
        degree = len(N.shape)
        func = self._get_func_second_kind(degree=degree)
        return func(N)

    def _get_func_second_kind(self, degree):
        funcs = {
            2: self.second_kind_N2,
            4: self.second_kind_N4,
        }
        return funcs[degree]

    def second_kind_N2(self, N):
        return 15.0 / 2.0 * (N - 1.0 / 5.0 * self._I2)

    def second_kind_N4(self, N):
        return (
            315.0
            / 8.0
            * (
                N
                - 2.0
                / 3.0
                * mechmean.operators.sym(
                    np.multiply.outer(self._I2, self.first_kind(degree=2))
                )
                + 1.0
                / 21.0
                * mechmean.operators.sym(np.multiply.outer(self._I2, self._I2))
            )
        )

    def third_kind(self, N):
        degree = len(N.shape)
        func = self._get_func_third_kind(degree=degree)
        return func(N)

    def _get_func_third_kind(self, degree):
        funcs = {2: self.third_kind_N2, 4: self.third_kind_N4}
        return funcs[degree]

    def third_kind_N2(self, N):
        return 15.0 / 2.0 * (N - 1.0 / 3.0 * self._I2)

    def third_kind_N4(self, N):
        return (
            315.0
            / 8.0
            * (
                N
                - 6.0
                / 7.0
                * mechmean.operators.sym(
                    np.multiply.outer(self._I2, self.first_kind(degree=2))
                )
                + 3.0
                / 35.0
                * mechmean.operators.sym(np.multiply.outer(self._I2, self._I2))
            )
        )


def evenly_distributed_vectors_on_sphere(nbr_vectors=1000):
    """
    Define nbr_vectors evenly distributed vectors on a sphere

    Using the golden spiral method kindly provided by
    stackoverflow-user "CR Drost"
    https://stackoverflow.com/a/44164075/8935243
    """
    from numpy import pi, cos, sin, arccos, arange

    indices = arange(0, nbr_vectors, dtype=float) + 0.5

    phi = arccos(1 - 2 * indices / nbr_vectors)
    theta = pi * (1 + 5 ** 0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    orientations = np.column_stack((x, y, z))
    return orientations


def first_kind_discrete(orientations, order=4):
    """
    Calc orientation tensors of ... kind
    """
    # Normalize orientations
    orientations = [np.array(v) / np.linalg.norm(v) for v in orientations]

    # Symmetrize orientations
    #    orientations_reversed = [-v for v in orientations]
    #    orientations = orientations + orientations_reversed

    einsumStrings = {
        1: "ij             -> j",
        2: "ij, ik         -> jk",
        3: "ij, ik, il     -> jkl",
        4: "ij, ik, il, im -> jklm",
        5: "ij, ik, il, im, in     -> jklmn",
        6: "ij, ik, il, im, in, ip -> jklmnp",
    }

    if order > 6:
        einsumStrings[order] = einsum_str_fabric_tensor_first_kind_discrete(order=order)

    einsumArgs = [orientations for i in range(order)]

    N = 1.0 / len(orientations) * np.einsum(einsumStrings[order], *einsumArgs)
    return N


def einsum_str_fabric_tensor_first_kind_discrete(order):
    """
    Generalize to higher orders:

    N = sum_i 'order'-times_dyad_product(vector)
    =
    1:  'ij             -> j',
    2:  'ij, ik         -> jk',
    3:  'ij, ik, il     -> jkl',
    4:  'ij, ik, il, im -> jklm',
    5:  'ij, ik, il, im, in     -> jklmn',
    6:  'ij, ik, il, im, in, ip -> jklmnp',
    ...
    """

    # Get list of all available characters
    import string

    letters = list(string.ascii_letters)
    letters.remove("i")

    # Create einsum string and arguments
    einsumInput = ",".join(["i" + letters[index] for index in range(order)])
    einsumOut = "".join(letters[0:order])
    einsumString = einsumInput + "->" + einsumOut

    return einsumString
