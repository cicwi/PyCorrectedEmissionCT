#!/usr/bin/env python

"""
Test `corrct.solvers` package.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from corrct import processing, projectors, solvers, regularizers


eps = np.finfo(np.float32).eps
OVERSIZE_DATA = 5

VOL_SHAPE_2D = np.asarray((10, 10))
PRJ_SHAPE_2D = np.asarray((OVERSIZE_DATA, np.prod(VOL_SHAPE_2D)))

TOLERANCE = 1e-2


def _print_max_deviation(test_ind: str, sol_diff: NDArray[np.floating], tolerance: float) -> None:
    """
    Print the maximum absolute deviation between the solution and the reference volume.

    Parameters
    ----------
    test_ind : str
        The identifier for the test.
    sol_diff : NDArray[np.floating]
        The difference between the solution and the reference volume.
    tolerance : float
        The tolerance level for the deviation.
    """
    max_dev = np.max(np.abs(sol_diff))
    print(f"\n{test_ind} - Max absolute deviation is: {max_dev:.6} (tolerance: {tolerance:.6}) -> ", end="", flush=True)


def _fwd_op(M: NDArray, x: NDArray, y_shape: tuple[int, int]) -> NDArray:
    """
    Perform the forward operation using the projection matrix.

    Parameters
    ----------
    M : NDArray
        The projection matrix.
    x : NDArray
        The input volume.
    y_shape : tuple[int, int]
        The shape of the output data.

    Returns
    -------
    NDArray
        The projected data.
    """
    return np.dot(M, x.flatten()).reshape(y_shape)


# def _bwd_op(M: NDArray, y: NDArray, x_shape: tuple[int, int]) -> NDArray:
#     return np.dot(y.flatten(), M).reshape(x_shape)


@pytest.fixture(scope="module")
def rand_2d_data() -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate random 2D data for testing.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The projection matrix, the random volume, and the projected data.
    """
    vol_rand_2d = np.fmin(np.random.rand(*VOL_SHAPE_2D[:2]) + eps, 1)
    proj_matrix_2d = (np.random.rand(int(np.prod(PRJ_SHAPE_2D)), int(np.prod(VOL_SHAPE_2D))) > 0.5).astype(np.float32)
    data_rand_2d = _fwd_op(proj_matrix_2d, vol_rand_2d, tuple(PRJ_SHAPE_2D))
    return proj_matrix_2d, vol_rand_2d, data_rand_2d


@pytest.fixture(scope="module")
def flat_2d_data() -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate flat 2D object data for testing.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        The projection matrix, the flat volume, and the projected data.
    """
    vol_flat_2d = processing.circular_mask(VOL_SHAPE_2D[:2], -2)
    proj_matrix_2d = (np.random.rand(int(np.prod(PRJ_SHAPE_2D)), int(np.prod(VOL_SHAPE_2D))) > 0.5).astype(np.float32)
    data_flat_2d = _fwd_op(proj_matrix_2d, vol_flat_2d, tuple(PRJ_SHAPE_2D))
    data_flat_2d += np.random.randn(*data_flat_2d.shape) * 1e-3
    return proj_matrix_2d, vol_flat_2d, data_flat_2d


@pytest.mark.parametrize("algo_it", [(solvers.SIRT, 1_000), (solvers.PDHG, 500), (solvers.MLEM, 10_000)])
def test_algo_rand(rand_2d_data, algo_it: tuple[type[solvers.Solver], int]):
    """
    Test algorithms in 2D on a random image.

    Parameters
    ----------
    rand_2d_data : tuple[NDArray, NDArray, NDArray]
        The random 2D data.
    algo_it : tuple[type[solvers.Solver], int]
        The algorithm class and the number of iterations.
    """
    proj_matrix_2d, vol_rand_2d, data_rand_2d = rand_2d_data
    algo_class, iterations = algo_it

    prj_mat = projectors.ProjectorMatrix(proj_matrix_2d, VOL_SHAPE_2D, PRJ_SHAPE_2D)

    algo = algo_class()
    sol, _ = algo(prj_mat, data_rand_2d, iterations=iterations)

    sol_diff = vol_rand_2d - sol
    _print_max_deviation(algo.info(), sol_diff, TOLERANCE)

    assert np.all(np.isclose(sol, vol_rand_2d, atol=TOLERANCE))


FLAT_2D_CASES = [
    (solvers.SIRT, 2_500, {}),
    (solvers.PDHG, 500, {}),
    (solvers.PDHG, 1_000, dict(precondition=False)),
    (solvers.PDHG, 500, dict(lower_limit=0, upper_limit=1)),
    (solvers.MLEM, 10_000, dict(lower_limit=0, upper_limit=1)),
]


@pytest.mark.parametrize("algo_it_pars", FLAT_2D_CASES)
def test_algo_flat(flat_2d_data, algo_it_pars: tuple[type[solvers.Solver], int, dict]):
    """
    Test algorithms in 2D on a flat 2D object.

    Parameters
    ----------
    flat_2d_object_data : tuple[NDArray, NDArray, NDArray]
        The flat 2D object data.
    algo_it_pars : tuple[type[solvers.Solver], int, dict]
        The algorithm class, the number of iterations, and extra parameters.
    """
    proj_matrix_2d, vol_flat_2d, data_flat_2d = flat_2d_data
    algo_class, iterations, extra_params = algo_it_pars

    prj_mat = projectors.ProjectorMatrix(proj_matrix_2d, VOL_SHAPE_2D, PRJ_SHAPE_2D)

    if algo_class in (solvers.SIRT, solvers.PDHG):
        reg = regularizers.Regularizer_TV2D(1e-4)
        algo = algo_class(regularizer=reg)
    else:
        algo = algo_class()
    sol, _ = algo(prj_mat, data_flat_2d, iterations=iterations, **extra_params)

    sol_diff = vol_flat_2d - sol
    _print_max_deviation(algo.info(), sol_diff, TOLERANCE)

    assert np.all(np.isclose(sol, vol_flat_2d, atol=TOLERANCE))
