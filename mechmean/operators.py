import itertools
import numpy as np
import mechkit


def sym(tensor, sym_axes=None):
    """Symmetrize selected axes of tensor

    If no sym_axes are specified, all axes are symmetrized
    """
    base_axis = np.array(range(len(tensor.shape)))

    sym_axes = base_axis if sym_axes is None else sym_axes

    perms = itertools.permutations(sym_axes)

    axes = list()
    for perm in perms:
        axis = base_axis.copy()
        axis[sym_axes] = perm
        axes.append(axis)

    return 1.0 / len(axes) * sum(tensor.transpose(axis) for axis in axes)


def sym_inner_mandel(tensor):
    con = mechkit.notation.Converter()
    t_mandel = con.to_mandel6(tensor)
    return con.to_like(
        inp=0.5 * (t_mandel + t_mandel.transpose()),
        like=tensor,
    )
