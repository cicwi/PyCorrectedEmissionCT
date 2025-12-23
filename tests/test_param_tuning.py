from os import getenv
from typing import Literal

import numpy as np
import pytest
import skimage.data as skd
import skimage.transform as skt
from numpy.typing import NDArray

import corrct as cct

SKIP_TESTS = getenv("GITHUB_ACTIONS") is not None


def iso_tv_seminorm(x):
    """
    Compute the isotropic TV seminorm of an image.

    Parameters
    ----------
    x : NDArray
        The input image.

    Returns
    -------
    float
        The isotropic TV seminorm of the image.
    """
    op = cct.operators.TransformGradient(x.shape)
    d = op(x)
    d = np.linalg.norm(d, axis=0, ord=2)
    return np.linalg.norm(d.flatten(), ord=1)


@pytest.fixture
def phantom() -> NDArray:
    """
    Fixture to create a phantom image using the Shepp-Logan phantom.

    Returns
    -------
    NDArray
        A downsampled Shepp-Logan phantom image.
    """
    ph: NDArray = skd.shepp_logan_phantom()
    ph = ph.astype(np.float32)
    return skt.downscale_local_mean(ph, 4)


@pytest.fixture
def sinogram_data(phantom: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """
    Fixture to generate sinogram data from the phantom image.

    Parameters
    ----------
    phantom : NDArray
        The phantom image.

    Returns
    -------
    tuple
        A tuple containing the sinogram, angles, and the expected phantom image.
    """
    ph, _, _ = cct.testing.phantom_assign_concentration(phantom)
    sinogram, angles, expected_ph, background_avg = cct.testing.create_sino(
        ph, 30, add_poisson=True, dwell_time_s=2e-2, background_avg=1e-2, background_std=1e-4
    )
    sinogram -= background_avg  # Subtract background_avg directly
    return sinogram, angles, expected_ph  # Remove background_avg from the output


def generate_task_init(reg=cct.regularizers.Regularizer_TV2D):
    """
    Generate a solver spawning function.

    Parameters
    ----------
    reg : callable
        A regularizer class.

    Returns
    -------
    callable
        A function that spawns a solver with a given lambda value.
    """

    def solver_spawn(lam_reg):
        return cct.solvers.PDHG(verbose=True, regularizer=reg(lam_reg), leave_progress=False)

    return solver_spawn


def generate_task_exec(phantom, angles, sinogram):
    """
    Generate a solver calling function.

    Parameters
    ----------
    phantom : NDArray
        The phantom image.
    angles : NDArray
        The angles for the sinogram.
    sinogram : NDArray
        The sinogram data.

    Returns
    -------
    callable
        A function that calls a solver with given parameters.
    """

    def solver_call(solver, b_test_mask=None):
        vol_mask = cct.processing.circular_mask(phantom.shape)
        with cct.projectors.ProjectorUncorrected(phantom.shape, angles) as A:
            return solver(A, sinogram, 200, x_mask=vol_mask, lower_limit=0, b_test_mask=b_test_mask)

    return solver_call


@pytest.mark.skipif(SKIP_TESTS, reason="Test is long and heavy, so we skip in a github action.")
@pytest.mark.parametrize("parallel_eval", [True, False, 2])
@pytest.mark.parametrize("num_averages", [1, 3])
def test_cross_validation(
    phantom: NDArray, sinogram_data: tuple[NDArray, NDArray, NDArray], parallel_eval: int, num_averages: int
):
    """
    Test the cross-validation functionality.

    Parameters
    ----------
    phantom : NDArray
        The phantom image.
    sinogram_data : tuple
        A tuple containing the sinogram, angles, and the expected phantom image.
    parallel_eval : bool
        Whether to use parallel evaluation.
    num_averages : int
        The number of averages to use.
    """
    debug = False  # Set this to True to enable plotting

    sinogram, angles, _ = sinogram_data  # Remove background_avg from the unpacking

    hpt_cv = cct.param_tuning.CrossValidation(
        sinogram.shape, verbose=True, num_averages=num_averages, parallel_eval=parallel_eval, plot_result=debug
    )
    hpt_cv.task_init_function = generate_task_init()
    hpt_cv.task_exec_function = generate_task_exec(phantom, angles, sinogram)

    lams_reg = cct.param_tuning.get_lambda_range(1e-3, 1e1, num_per_order=2)
    f_avgs, f_stds, _ = hpt_cv.compute_loss_values(lams_reg)

    assert f_avgs.shape == (len(lams_reg),)
    assert f_stds.shape == (len(lams_reg),)


@pytest.mark.skipif(SKIP_TESTS, reason="Test is long and heavy, so we skip in a github action.")
@pytest.mark.parametrize("parallel_eval", [True, False, 2])
def test_reconstruction_error(phantom: NDArray, sinogram_data: tuple[NDArray, NDArray, NDArray], parallel_eval: int):
    """
    Test the reconstruction error functionality.

    Parameters
    ----------
    phantom : NDArray
        The phantom image.
    sinogram_data : tuple
        A tuple containing the sinogram, angles, and the expected phantom image.
    parallel_eval : bool
        Whether to use parallel evaluation.
    """
    debug = False  # Set this to True to enable plotting

    sinogram, angles, expected_ph = sinogram_data  # Remove background_avg from the unpacking

    hpt_cv = cct.param_tuning.CrossValidation(
        sinogram.shape, verbose=True, num_averages=3, parallel_eval=parallel_eval, plot_result=debug
    )
    hpt_cv.task_init_function = generate_task_init()
    hpt_cv.task_exec_function = generate_task_exec(phantom, angles, sinogram)

    lams_reg = cct.param_tuning.get_lambda_range(1e-3, 1e1, num_per_order=2)

    err_l1, err_l2 = hpt_cv.compute_reconstruction_error(lams_reg, expected_ph)

    assert err_l1.shape == (len(lams_reg),)
    assert err_l2.shape == (len(lams_reg),)


@pytest.mark.skipif(SKIP_TESTS, reason="Test is long and heavy, so we skip in a github action.")
@pytest.mark.parametrize("parallel_eval", [True, False, 2])
@pytest.mark.parametrize("use_two_function", [True, False])
def test_l_curve(
    phantom: NDArray, sinogram_data: tuple[NDArray, NDArray, NDArray], parallel_eval: int, use_two_function: bool
):
    """
    Test the L-curve functionality.

    Parameters
    ----------
    phantom : NDArray
        The phantom image.
    sinogram_data : tuple
        A tuple containing the sinogram, angles, and the expected phantom image.
    parallel_eval : bool
        Whether to use parallel evaluation.
    """
    debug = False  # Set this to True to enable plotting

    sinogram, angles, _ = sinogram_data  # Remove background_avg from the unpacking

    hpt_lc = cct.param_tuning.LCurve(iso_tv_seminorm, verbose=True, plot_result=debug, parallel_eval=parallel_eval)
    if use_two_function:
        hpt_lc.task_init_function = generate_task_init()
        hpt_lc.task_exec_function = generate_task_exec(phantom, angles, sinogram)
    else:

        def solve_reg(lam: float):
            solver = generate_task_init()(lam)
            return generate_task_exec(phantom, angles, sinogram)(solver)

        hpt_lc.task_exec_function = solve_reg

    lams_reg = cct.param_tuning.get_lambda_range(1e-3, 1e1, num_per_order=2)
    f_vals_lc = hpt_lc.compute_loss_values(lams_reg)

    assert f_vals_lc.shape == (len(lams_reg),)


if __name__ == "__main__":
    pytest.main()
