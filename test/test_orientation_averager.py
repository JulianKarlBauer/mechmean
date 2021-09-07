#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mechkit
import mechmean
from mechmean import material
from mechmean import orientation_averager_alternatives


def test_compare_AdvaniTucker_N_D():

    sym = mechmean.operators.sym
    con = mechkit.notation.Converter()

    for i in range(10):
        f = material.AlignedStiffnessFactory()
        B = f.positiv_definit(label="hexagonal_axis1")

        # B = np.array([[0.7327, 0.8   , 0.8   , 0.    , 0.    , 0.    ],
        #        [0.8   , 0.9987, 0.9116, 0.    , 0.    , 0.    ],
        #        [0.8   , 0.9116, 0.9987, 0.    , 0.    , 0.    ],
        #        [0.    , 0.    , 0.    , 0.1743, 0.    , 0.    ],
        #        [0.    , 0.    , 0.    , 0.    , 0.3402, 0.    ],
        #        [0.    , 0.    , 0.    , 0.    , 0.    , 0.3402]])

        rand = sym(np.random.rand(3, 3, 3, 3))
        N4 = con.to_mandel6(rand) - 1.0 / 6.0 * (
            np.einsum("ijij->", rand) - 1.0
        ) * np.eye(6, 6)

        kanatani = mechmean.fabric_tensors.KanataniFactory(N=N4)
        D2 = kanatani.D2
        D4 = kanatani.D4

        at_D = orientation_averager_alternatives.AdvaniTucker_in_kanatani_third_kind(
            D2, D4
        )
        at = mechmean.orientation_averager.AdvaniTucker(N4)

        av_D = at_D.average(B)
        av = at.average(B)

        for i in range(5):
            print("N_base_{}".format(i))
            print(con.to_mandel9(at.base[i, :, :]))
            print("D_base_{}".format(i))
            print(con.to_mandel9(at_D.base[i, :, :]))

        print("av_D")
        print(av_D)

        print("av")
        print(av)

        print("av-av_D")
        print(av - av_D)

        assert np.allclose(av, av_D)
