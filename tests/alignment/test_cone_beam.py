import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from corrct.alignment.cone_beam import FitConeBeamGeometry, ConeBeamGeometry
from corrct.models import ProjectionGeometry, plot_projection_geometry


def project_to_uv(points_xyz: NDArray, proj_geom: ProjectionGeometry) -> NDArray:
    """
    Project a list of XYZ points to UV coordinates using precomputed geometry.

    Parameters
    ----------
    points_xyz : NDArray
        Array of (x, y, z) points in the sample volume (XYZ coordinates).
    geo : ProjectionGeometry
        ProjectionGeometry class with precomputed geometry properties.

    Returns
    -------
    NDArray
        Array of (u, v) coordinates for each input point.
    """
    points_xyz = np.array(points_xyz, dtype=np.float32)

    det_u_xyz = proj_geom.det_u_xyz.flatten()
    det_v_xyz = proj_geom.det_v_xyz.flatten()

    det_normal_xyz = np.cross(det_u_xyz, det_v_xyz)

    vecs_src_2_pnts = points_xyz - proj_geom.src_pos_xyz
    vecs_src_2_pnts = vecs_src_2_pnts / np.linalg.norm(vecs_src_2_pnts, axis=-1, keepdims=True)

    denominators = np.dot(det_normal_xyz, vecs_src_2_pnts.T)
    numerators = (proj_geom.det_pos_xyz - points_xyz).dot(det_normal_xyz)

    invalid_points = np.isclose(denominators, 0.0)
    valid_denoms = np.logical_not(invalid_points)

    lams = numerators[list(valid_denoms)] / denominators[list(valid_denoms)]
    prj_pnts_xyz = vecs_src_2_pnts[list(valid_denoms), :] * lams[:, None] + points_xyz[list(valid_denoms), :]

    uv_coords = np.empty((len(points_xyz), 2), dtype=np.float32)
    uv_coords.fill(np.nan)
    uv_coords[list(valid_denoms), :] = np.stack((prj_pnts_xyz.dot(det_u_xyz), prj_pnts_xyz.dot(det_v_xyz)), axis=1)

    return uv_coords


def compute_xyz_rotated(
    r: float, z: float, ws: NDArray, voxel_size_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> NDArray:
    """
    Compute XYZ coordinates for an array of (r, z, w) inputs in the rotated volume.

    Parameters
    ----------
    r : float
        Radial distance from volume center.
    z : float
        Elevation (Z coordinates).
    ws : NDArray
        Array of rotation angles (omega) in radians.
    voxel_size_xyz : tuple[float, float, float], optional
        Voxel sizes in X, Y, Z directions (default: (1.0, 1.0, 1.0)).

    Returns
    -------
    NDArray
        Array of (x, y, z) coordinates in the rotated volume.
    """
    ws = np.squeeze(np.array(ws))

    # Check if ws is one-dimensional
    if ws.ndim != 1:
        raise ValueError("Input array ws must be one-dimensional.")

    # At w=0, the point is along Y: (0, r, z)
    # Rotation matrix for w (omega) around Z:
    # [cos(w), -sin(w), 0]
    # [sin(w),  cos(w), 0]
    # [0,       0,      1]
    rot_w = Rotation.from_rotvec(np.array([np.zeros_like(ws), np.zeros_like(ws), ws]).T)

    rot_xyz = rot_w.apply(np.array([0, r, z]).T)

    # Scale by voxel sizes
    return rot_xyz * np.array(voxel_size_xyz)


@pytest.mark.parametrize("R", [30.0, 50.0])
@pytest.mark.parametrize("D", [70.0, 80.0])
def test_cone_beam_ellipse_distances(R: float, D: float):
    debug = False

    det_size_vu = 12

    acq_geom_tst = ConeBeamGeometry(theta_deg=0.0, phi_deg=0.0, eta_deg=0.0, D_pix=D, R_pix=R)
    proj_geom_tst = acq_geom_tst.get_prj_geom()
    if debug:
        print(f"{acq_geom_tst = }")
        plot_projection_geometry(proj_geom_tst, acq_geom_tst.get_vol_geom())

    r = det_size_vu / 2 - 1
    z_u = -2.5
    z_l = 1.5
    ws = np.deg2rad(np.arange(0, 360, 6))
    points_xyz_u = compute_xyz_rotated(r=r, z=z_u, ws=ws)
    points_xyz_l = compute_xyz_rotated(r=r, z=z_l, ws=ws)

    # fig, axs = plt.subplots(1, 1)
    # axs.plot(points_xyz[:, 0], points_xyz[:, 1])
    # axs.grid()
    # fig.tight_layout()
    # plt.show()

    prj_uv_coords_u = project_to_uv(points_xyz_u, proj_geom_tst)
    prj_uv_coords_l = project_to_uv(points_xyz_l, proj_geom_tst)

    # fig, axs = plt.subplots(1, 1)
    # axs.plot(prj_uv_coords_u[:, 0], prj_uv_coords_u[:, 1])
    # axs.plot(prj_uv_coords_l[:, 0], prj_uv_coords_l[:, 1])
    # axs.grid()
    # fig.tight_layout()
    # plt.show()

    prj_size_vu = np.array((det_size_vu, det_size_vu))

    points_ell1 = np.flip(prj_uv_coords_u.T, axis=0) + prj_size_vu[:, None] / 2
    points_ell2 = np.flip(prj_uv_coords_l.T, axis=0) + prj_size_vu[:, None] / 2

    if debug:
        print(f"{points_ell1 = }")
        print(f"{points_ell2 = }")

    geom_fit = FitConeBeamGeometry(prj_size_vu=prj_size_vu, points_ell1=points_ell1, points_ell2=points_ell2)
    geom_fit.fit(r=r, e=1.0)

    if debug:
        print(f"{geom_fit.acq_geom = }")

    assert np.isclose(
        D, geom_fit.acq_geom.D_pix, rtol=1e-5, atol=1e-3
    ), f"The fitted source-detector distance is wrong ({geom_fit.acq_geom.D_pix:.6} vs expected: {D:.6})"
    assert np.isclose(
        R, geom_fit.acq_geom.R_pix, rtol=1e-5, atol=1e-3
    ), f"The fitted source-sample distance is wrong ({geom_fit.acq_geom.R_pix:.6} vs expected: {R:.6})"


@pytest.mark.parametrize(("p", "n"), [(30.0, 0.0), (5.0, 1.0), (0.0, 15.0)])
def test_cone_beam_ellipse_angles(p: float, n: float):
    debug = False

    det_size_vu = 12

    acq_geom_tst = ConeBeamGeometry(theta_deg=0.0, phi_deg=p, eta_deg=n, D_pix=80.0, R_pix=50.0)
    proj_geom_tst = acq_geom_tst.get_prj_geom()
    if debug:
        print(f"{acq_geom_tst = }")
        print(f"{proj_geom_tst = }")
        plot_projection_geometry(proj_geom_tst, acq_geom_tst.get_vol_geom())

    r = det_size_vu / 2 - 1
    z_u = -2.5
    z_l = 1.5
    ws = np.deg2rad(np.arange(0, 360, 6))
    points_xyz_u = compute_xyz_rotated(r=r, z=z_u, ws=ws)
    points_xyz_l = compute_xyz_rotated(r=r, z=z_l, ws=ws)

    # fig, axs = plt.subplots(1, 1)
    # axs.plot(points_xyz[:, 0], points_xyz[:, 1])
    # axs.grid()
    # fig.tight_layout()
    # plt.show()

    prj_uv_coords_u = project_to_uv(points_xyz_u, proj_geom_tst)
    prj_uv_coords_l = project_to_uv(points_xyz_l, proj_geom_tst)

    # fig, axs = plt.subplots(1, 1)
    # axs.plot(prj_uv_coords_u[:, 0], prj_uv_coords_u[:, 1])
    # axs.plot(prj_uv_coords_l[:, 0], prj_uv_coords_l[:, 1])
    # axs.grid()
    # fig.tight_layout()
    # plt.show()

    prj_size_vu = np.array((det_size_vu, det_size_vu))

    points_ell1 = np.flip(prj_uv_coords_u.T, axis=0) + prj_size_vu[:, None] / 2
    points_ell2 = np.flip(prj_uv_coords_l.T, axis=0) + prj_size_vu[:, None] / 2

    if debug:
        print(f"{points_ell1 = }")
        print(f"{points_ell2 = }")

    geom_fit = FitConeBeamGeometry(prj_size_vu=prj_size_vu, points_ell1=points_ell1, points_ell2=points_ell2)
    geom_fit.fit(r=r, e=1.0)

    if debug:
        print(f"{geom_fit.acq_geom = }")

    assert np.isclose(
        p, geom_fit.acq_geom.phi_deg, rtol=5e-2, atol=5e-1
    ), f"The fitted phi angle is wrong ({geom_fit.acq_geom.phi_deg:.6} vs expected: {p:.6})"
    assert np.isclose(
        n, geom_fit.acq_geom.eta_deg, rtol=1e-2, atol=1e-1
    ), f"The fitted yaw angle is wrong ({geom_fit.acq_geom.eta_deg:.6} vs expected: {n:.6})"


if __name__ == "__main__":
    points_xyz_tst = np.array([[0, 1, 2], [3, 4, 5]])

    # acq_geom = ConeBeamGeometry(theta_deg=0.0, phi_deg=0.0, eta_deg=0.0, D_pix=260.0, R_pix=160.0)
    acq_geom = ConeBeamGeometry(
        theta_deg=0.0,
        phi_deg=0.0,
        eta_deg=np.rad2deg(np.pi / 4),
        D_pix=260.0,
        R_pix=160.0,
        det_size_u_pix=100,
        det_size_v_pix=100,
    )
    print(acq_geom)
    prj_geom = acq_geom.get_prj_geom()
    prj_uv_coords = project_to_uv(points_xyz=points_xyz_tst, proj_geom=prj_geom)
    print(prj_uv_coords)

    # Compute rotated XYZ for arrays of (r, z, w)
    ws = np.array([0, np.pi / 4, np.pi / 2])
    xyz_rotated = compute_xyz_rotated(r=2.0, z=1.0, ws=ws)
    print(xyz_rotated)

    plot_projection_geometry(acq_geom.get_prj_geom(), acq_geom.get_vol_geom())
