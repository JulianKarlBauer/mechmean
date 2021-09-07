#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mechkit

import mechmean


def test_fabric_tensor_first_kind_discrete():
    """Compare einsum-implementation with loop-implementation"""

    orientations = np.random.rand(10, 3)  # Ten random vectors in 3D

    # Normalize orientations
    orientations = [np.array(v) / np.linalg.norm(v) for v in orientations]

    # Symmetrize orientations
    orientations_reversed = [-v for v in orientations]
    orientations = orientations + orientations_reversed

    def oT_loops(orientations, order=4):
        N = np.zeros((3,) * order)
        for p in orientations:
            out = p
            for index in range(order - 1):
                out = np.multiply.outer(out, p)
            N[:] = N[:] + out
        N = N / len(orientations)
        return N

    for order in range(1, 10):
        assert np.allclose(
            mechmean.fabric_tensors.first_kind_discrete(
                order=order,
                orientations=orientations,
            ),
            oT_loops(
                order=order,
                orientations=orientations,
            ),
        )


def test_fabric_tensor_first_kind_discrete_benchmarks():
    """Compare with known benchmarks from presentation M. Schneider 26.09.2018"""

    orientations = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    N2_ref = np.zeros((3, 3))
    N2_ref[0, 0] = N2_ref[1, 1] = N2_ref[2, 2] = 1.0 / 3.0

    N4_ref = np.zeros((6, 6))
    N4_ref[0, 0] = N4_ref[1, 1] = N4_ref[2, 2] = 1.0 / 3.0
    converter = mechkit.notation.Converter()
    N4_ref_t = converter.to_tensor(N4_ref)

    f = mechmean.fabric_tensors.first_kind_discrete

    assert np.allclose(
        N2_ref,
        f(
            order=2,
            orientations=orientations,
        ),
    )

    assert np.allclose(
        N4_ref_t,
        f(
            order=4,
            orientations=orientations,
        ),
    )


# ##########################################################################
# ##########################    Trivial tests     ##########################
# ##########################################################################


def test_N2_iso():
    """This test is trivial. The contraction is a
    matrix-vector multiplication with vector [1, 1, 1, 0, 0, 0]
    which results in 1/5 * (3/3 + 1/3 + 1/3) = 1/3
    """

    con = mechkit.notation.Converter()

    N4_iso = (
        1.0
        / 5.0
        * np.array(
            [
                [1.0, 1.0 / 3.0, 1.0 / 3.0, 0.0, 541.0, 0.0],
                [1.0 / 3.0, 1.0, 1.0 / 3.0, 0.0, 0.0, 0.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0, 6874.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0 / 3.0, 0.0, 34345.0],
                [0.0, 0.0, 0.0, 34.0, 3434.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0 / 3.0],
            ]
        )
    )

    N2_iso = 1.0 / 3.0 * np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    contracted = np.einsum(
        "ijkl, kl->ij",
        con.to_tensor(N4_iso),
        mechkit.tensors.Basic().I2,
    )

    assert np.allclose(N2_iso, contracted)


def test_N2_planar_iso():
    """This test is trivial. The contraction is a
    matrix-vector multiplication with vector [1, 1, 1, 0, 0, 0]
    """

    con = mechkit.notation.Converter()

    N4_planar_iso = (
        1.0
        / 8.0
        * np.array(
            [
                [3.0, 1.0, 0.0, 0.0, 45333.0, 0.0],
                [1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 543.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 453354.0, 0.0],
                [0.0, 0.0, 0.0, 453453.0, 0.0, 2.0],
            ]
        )
    )

    N2_planar_iso = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])

    contracted = np.einsum(
        "ijkl, kl->ij",
        con.to_tensor(N4_planar_iso),
        mechkit.tensors.Basic().I2,
    )

    assert np.allclose(N2_planar_iso, contracted)


if __name__ == "__main__":
    test_N2_planar_iso()
