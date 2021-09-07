#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import itertools
import mechkit

import mechmean


np.set_printoptions(
    linewidth=140,
    precision=2,
    # suppress=False,
)


def test_compare_Hill_implementations():
    def get_polarization_alternative_implementation(aspect_ratio, matrix):
        Km = matrix.K
        Gm = matrix.G
        ar = aspect_ratio

        arsqu = ar ** 2.0
        denom = Gm * (1.0 - arsqu) * (4.0 * Gm + 3.0 * Km)
        if ar < 1.0:
            print("ATTENTION: aspect ratio < 1 ..formulas might be incorrect!")
            h = (
                ar
                * (np.arccos(ar) - ar * np.sqrt(1.0 - arsqu))
                * (1.0 - arsqu) ** (-1.5)
            )
        else:
            h = (
                ar
                * (ar * np.sqrt(arsqu - 1.0) - np.arccosh(ar))
                / (arsqu - 1.0) ** (1.5)
            )

        P0k = (
            Gm * (7 * h - 2 * arsqu - 4 * arsqu * h)
            + 3 * Km * (h - 2 * arsqu + 2 * arsqu * h)
        ) / (8.0 * denom)
        P0m = ((Gm + 3 * Km) * (2 * arsqu - h - 2 * arsqu * h)) / (4.0 * denom)
        P0r = (
            Gm * (6 - 5 * h - 8 * arsqu + 8 * arsqu * h)
            + 3 * Km * (h - 2 * arsqu + 2 * arsqu * h)
        ) / (2.0 * denom)
        P0p = (
            Gm * (15 * h - 2 * arsqu - 12 * arsqu * h) + 3 * Km * (3 * h - 2 * arsqu)
        ) / (16.0 * denom)
        P0q = (
            2 * Gm * (4 - 3 * h - 2 * arsqu)
            + 3 * Km * (2 - 3 * h + 2 * arsqu - 3 * arsqu * h)
        ) / (8.0 * denom)

        P0 = np.array(
            [
                [P0r, P0m, P0m, 0, 0, 0],
                [P0m, P0k + P0p, P0k - P0p, 0, 0, 0],
                [P0m, P0k - P0p, P0k + P0p, 0, 0, 0],
                [0, 0, 0, 2 * P0p, 0, 0],
                [0, 0, 0, 0, 2 * P0q, 0],
                [0, 0, 0, 0, 0, 2 * P0q],
            ]
        )
        return P0

    factory = mechmean.hill_polarization.Factory()

    for K, G in [
        (1e6, 4e5),
        (1666.0, 769.0),
        (300, 120),
    ]:
        matrix = mechkit.material.Isotropic(K=K, G=G)
        for aspect_ratio in [0.4, 1.1, 10, 100]:
            # Hill Bauer
            P_ref = factory.spheroid_castaneda(aspect_ratio=aspect_ratio, matrix=matrix)
            # Hill alternative
            P_2 = get_polarization_alternative_implementation(
                aspect_ratio=aspect_ratio, matrix=matrix
            )
            assert np.allclose(P_ref, P_2)


def test_compare_Hill_polarization_Castaneda_Mura():

    factory = mechmean.hill_polarization.Factory()

    for K, G in [(1e6, 4e5), (1666.6, 769.3), (200, 100)]:
        matrix = mechkit.material.Isotropic(K=K, G=G)
        for aspect_ratio in [10, 1.5, 0.5, 0.1]:

            # Mura
            P_e = factory.spheroid_mura(aspect_ratio=aspect_ratio, matrix=matrix)

            # Castaneda
            P_h = factory.spheroid_castaneda(aspect_ratio=aspect_ratio, matrix=matrix)

            # Compare
            print("\nK = {}\nG = {}\naspect_ratio = {}".format(K, G, aspect_ratio))
            print("P_h == P_e is", np.allclose(P_h, P_e))

            assert np.allclose(P_h, P_e)


def test_Hill_spheroid_check_tolerance_sphere_Mura():
    """As tol in this function is smaller than tol in decision-maker deciding
    between spheroid and sphere, this function ONLY calculates SPHERES."""

    factory = mechmean.hill_polarization.Factory()
    tol = 1e-5

    for K, G in [(1e6, 4e5), (2000, 1000)]:
        matrix = mechkit.material.Isotropic(K=K, G=G)
        P = []
        for aspect_ratio in [1.0, 1.0 - tol, 1.0 + tol]:
            P_tmp = factory.spheroid_mura(aspect_ratio=aspect_ratio, matrix=matrix)
            P.append(P_tmp)
        print("K=", K, "G=", G)
        print(P)
        combinations = list(itertools.combinations(P, 2))
        assert all([np.allclose(x, y) for x, y in combinations])


def test_Hill_spheroid_check_tolerance_sphere_Castaneda():
    """As tol in this function is smaller than tol in decision-maker deciding
    between spheroid and sphere, this function ONLY calculates SPHERES."""

    factory = mechmean.hill_polarization.Factory()
    tol = 1e-5

    for K, G in [(1e6, 4e5), (2000, 1000)]:
        matrix = mechkit.material.Isotropic(K=K, G=G)
        P = []
        for aspect_ratio in [1.0, 1.0 - tol, 1.0 + tol]:
            P_tmp = factory.spheroid_castaneda(aspect_ratio=aspect_ratio, matrix=matrix)
            P.append(P_tmp)
        print("K=", K, "G=", G)
        print(P)
        combinations = list(itertools.combinations(P, 2))
        assert all([np.allclose(x, y) for x, y in combinations])


def test_compare_Hill_Castaneda_Mura_sphere():

    factory = mechmean.hill_polarization.Factory()

    aspect_ratio = 1.0
    for K, G in [(1e6, 4e5), (1666.6, 769.3), (200, 100)]:
        matrix = mechkit.material.Isotropic(K=K, G=G)

        # Mura
        P_Mura = factory.spheroid_mura(aspect_ratio=aspect_ratio, matrix=matrix)

        # Castaneda
        P_Casta = factory.spheroid_castaneda(aspect_ratio=aspect_ratio, matrix=matrix)

        # Compare
        print("\nK = {}\nG = {}\naspect_ratio = {}".format(K, G, aspect_ratio))
        print("P_Casta == P_Mura is", np.allclose(P_Casta, P_Mura))
        print(P_Casta - P_Mura)

        assert np.allclose(P_Casta, P_Mura)


def test_compare_Hill_needle_as_limit_spheroid():

    factory = mechmean.hill_polarization.Factory()

    aspect_ratio = 10000.0
    for K, G in [(1e6, 4e5), (1666.6, 769.3), (200, 100)]:
        matrix = mechkit.material.Isotropic(K=K, G=G)

        # Mura
        P_Mura = factory.spheroid_mura(aspect_ratio=aspect_ratio, matrix=matrix)

        # Castaneda
        P_Casta = factory.spheroid_castaneda(aspect_ratio=aspect_ratio, matrix=matrix)

        # Needle
        P_needle = mechmean.hill_polarization.Castaneda().needle(matrix=matrix)

        # Compare
        assert np.allclose(P_needle, P_Mura)
        assert np.allclose(P_needle, P_Casta)


def test_compare_Hill_needle_Castaneda_Ortolano():

    for K, G in [(1e6, 4e5), (1666.6, 769.3), (200, 100)]:
        matrix = mechkit.material.Isotropic(K=K, G=G)

        P_Ortolano = mechmean.hill_polarization.Ortolano().needle(matrix=matrix)

        P_Casta = mechmean.hill_polarization.Castaneda().needle(matrix=matrix)

        # Compare
        assert np.allclose(P_Casta, P_Ortolano)


def test_compare_Hill_spheroid_Castaneda_Mura():
    factory = mechmean.hill_polarization.Factory()
    for aspect_ratio in [0.1, 0.5, 1.0, 1.5, 3, 10, 100]:
        for K, G in [(1e6, 4e5), (1666.6, 769.3), (200, 100)]:
            matrix = mechkit.material.Isotropic(K=K, G=G)

            # Mura
            P_Mura = factory.spheroid_mura(aspect_ratio=aspect_ratio, matrix=matrix)

            # Castaneda
            P_Casta = factory.spheroid_castaneda(
                aspect_ratio=aspect_ratio, matrix=matrix
            )

            # Compare
            assert np.allclose(P_Mura, P_Casta)


if __name__ == "__main__":

    factory = mechmean.hill_polarization.Factory()

    for K, G in [
        (1e6, 4e5),
    ]:
        matrix = mechkit.material.Isotropic(K=K, G=G)
        for aspect_ratio in [1.0]:

            # Mura
            P_Mura = factory.spheroid_mura(aspect_ratio=aspect_ratio, matrix=matrix)

            # Castaneda
            P_Kehrer = factory.spheroid_castaneda(
                aspect_ratio=aspect_ratio, matrix=matrix
            )

            printQueue = [
                "P_Mura",
                "P_Kehrer",
                "np.allclose(P_Mura, P_Kehrer)",
            ]
            # Print
            for val in printQueue:
                print(val)
                print(eval(val), "\n")
