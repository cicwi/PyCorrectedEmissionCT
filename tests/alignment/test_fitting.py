import numpy as np
import pytest
from numpy.typing import NDArray

from corrct.alignment.fitting import Ellipse, fit_ellipse_center, fit_ellipse_parameters


def generate_points(ellipse: Ellipse, num_points: int, noise_std: float = 0.0, outliers: int = 0) -> NDArray:
    """Generate points on the ellipse with optional noise and outliers.

    Parameters
    ----------
    ellipse : Ellipse
        The ellipse to generate points from.
    num_points : int
        Number of points to generate.
    noise : float, optional
        Standard deviation of the Gaussian noise to add to the points. Default is 0.0.
    outliers : int, optional
        Number of outliers to add to the points. Default is 0.

    Returns
    -------
    NDArray
        Generated points in VU coordinates.
    """
    angles = np.linspace(0, np.pi, num_points)
    x = np.cos(angles) / np.sqrt(ellipse.a) + ellipse.u
    y = ellipse(x)

    points = np.stack([[*y[0], *np.flip(y[1])], [*x, *np.flip(x)]], axis=0)

    if noise_std > 0.0:
        points += np.random.normal(scale=noise_std, size=points.shape)

    if outliers > 0:
        outlier_indices = np.random.choice(num_points, size=outliers, replace=False)
        points[:, outlier_indices] += np.random.normal(scale=5 * noise_std, size=(2, outliers))

    # import matplotlib.pyplot as plt

    # print(ellipse)
    # print(points)
    # fig, axs = plt.subplots(1, 1)
    # axs.plot(x, y[0])
    # axs.plot(x, y[1])
    # axs.scatter(points[1], points[0], color="C2")
    # fig.tight_layout()
    # plt.show()

    return points


def test_fit_ellipse_center_simple() -> None:
    """Test fit_ellipse_center with no noise and 6 points."""
    ellipse = Ellipse(a=2.0, b=1.5, c=0.5, u=1.0, v=1.0, c_vu=np.ones(2))
    points = generate_points(ellipse, num_points=6)

    center = fit_ellipse_center(points, rescale=True, use_l1_norm=False)

    assert np.allclose(center, np.array([[1.0], [1.0]]), atol=1e-2), "Center should be close to [1.0, 1.0]"


def test_fit_ellipse_center_noisy() -> None:
    """Test fit_ellipse_center with small Gaussian noise and 60 points."""
    ellipse = Ellipse(a=2.0, b=1.5, c=0.5, u=1.0, v=1.0, c_vu=np.ones(2))
    points = generate_points(ellipse, num_points=60, noise_std=0.1)

    center = fit_ellipse_center(points, rescale=True, use_l1_norm=False)

    assert np.allclose(center, np.array([[1.0], [1.0]]), atol=1e-1), "Center should be close to [1.0, 1.0]"


def test_fit_ellipse_center_noisy_with_outliers() -> None:
    """Test fit_ellipse_center with 60 points and 2 outliers using L1 norm."""
    ellipse = Ellipse(a=2.0, b=1.5, c=0.5, u=1.0, v=1.0, c_vu=np.ones(2))
    points = generate_points(ellipse, num_points=60, noise_std=0.1, outliers=2)

    center = fit_ellipse_center(points, rescale=True, use_l1_norm=True)

    assert np.allclose(center, np.array([[1.0], [1.0]]), atol=1e-1), "Center should be close to [1.0, 1.0]"


def test_fit_ellipse_parameters_simple() -> None:
    """Test fit_ellipse_parameters with no noise and 6 points."""
    ellipse = Ellipse(a=2.0, b=1.5, c=0.5, u=1.0, v=1.0, c_vu=np.ones(2))
    points = generate_points(ellipse, num_points=6)

    a, b, c, u, v = fit_ellipse_parameters(points, rescale=True, use_l1_norm=False)

    assert np.isclose(a, 2.0, atol=1e-2), "Semi-major axis (a) should be close to 2.0"
    assert np.isclose(b, 1.5, atol=1e-2), "Semi-minor axis (b) should be close to 1.5"
    assert np.isclose(c, 0.5, atol=1e-2), "Rotation parameter (c) should be close to 0.5"
    assert np.isclose(u, 1.0, atol=1e-2), "Center along the x-axis (u) should be close to 1.0"
    assert np.isclose(v, 1.0, atol=1e-2), "Center along the y-axis (v) should be close to 1.0"


def test_fit_ellipse_parameters_noisy() -> None:
    """Test fit_ellipse_parameters with small Gaussian noise and 60 points."""
    ellipse = Ellipse(a=2.0, b=1.5, c=0.5, u=1.0, v=1.0, c_vu=np.ones(2))
    points = generate_points(ellipse, num_points=60, noise_std=0.02)

    a, b, c, u, v = fit_ellipse_parameters(points, rescale=True, use_l1_norm=False)

    assert np.isclose(a, 2.0, atol=1e-1), "Semi-major axis (a) should be close to 2.0"
    assert np.isclose(b, 1.5, atol=1e-1), "Semi-minor axis (b) should be close to 1.5"
    assert np.isclose(c, 0.5, atol=1e-1), "Rotation parameter (c) should be close to 0.5"
    assert np.isclose(u, 1.0, atol=1e-1), "Center along the x-axis (u) should be close to 1.0"
    assert np.isclose(v, 1.0, atol=1e-1), "Center along the y-axis (v) should be close to 1.0"


def test_fit_ellipse_parameters_noisy_with_outliers() -> None:
    """Test fit_ellipse_parameters with 60 points and 2 outliers using L1 norm."""
    ellipse = Ellipse(a=2.0, b=1.5, c=0.5, u=1.0, v=1.0, c_vu=np.ones(2))
    points = generate_points(ellipse, num_points=60, noise_std=0.02, outliers=2)

    a, b, c, u, v = fit_ellipse_parameters(points, rescale=True, use_l1_norm=True)

    assert np.isclose(a, 2.0, atol=1e-1), "Semi-major axis (a) should be close to 2.0"
    assert np.isclose(b, 1.5, atol=1e-1), "Semi-minor axis (b) should be close to 1.5"
    assert np.isclose(c, 0.5, atol=1e-1), "Rotation parameter (c) should be close to 0.5"
    assert np.isclose(u, 1.0, atol=1e-1), "Center along the x-axis (u) should be close to 1.0"
    assert np.isclose(v, 1.0, atol=1e-1), "Center along the y-axis (v) should be close to 1.0"
