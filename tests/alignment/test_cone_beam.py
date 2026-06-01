from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from corrct.alignment.cone_beam import FitConeBeamGeometry


@dataclass
class DetectorGeometry:
    det_u_xyz: NDArray
    det_v_xyz: NDArray
    det_center_xyz: NDArray
    src_pos_xyz: NDArray
    u0: float
    v0: float
    pu: float
    pv: float


def compute_detector_geometry(
    R: float,
    d: Sequence[float] | NDArray,
    p: float,
    t: float,
    n: float,
    pu: float,
    pv: float,
    u_sign: int = 1,
    v_sign: int = -1,
) -> DetectorGeometry:
    """
    Compute the detector geometry (u, v, center, u0, v0) from the given parameters.

    Parameters:
    - R: Distance from source to volume center.
    - d: Vector (dx, dy, dz) from volume center to detector center in XYZ.
    - p, t, n: Detector tilts (phi, theta, eta) in radians.
    - pu, pv: Pixel sizes in U and V directions.
    - u_sign, v_sign: Sign convention for U and V axes (default: 1, -1).

    Returns:
    - DetectorGeometry dataclass with computed properties.
    """
    d = np.array(d)
    src_pos = np.array([-R, 0, 0])
    det_center = d

    # Apply tilts in order: p (phi), t (theta), n (eta)
    # p (phi) around Z:
    # [cos(p), -sin(p), 0]
    # [sin(p),  cos(p), 0]
    # [0,       0,      1]
    rot_p = Rotation.from_rotvec([0, 0, p])
    # t (theta) around Y:
    # [cos(t),  0, sin(t)]
    # [0,       1, 0     ]
    # [-sin(t), 0, cos(t)]
    rot_t = Rotation.from_rotvec([0, t, 0])
    # n (eta) around X:
    # [1,      0,       0]
    # [0, cos(n), -sin(n)]
    # [0, sin(n),  cos(n)]
    rot_n = Rotation.from_rotvec([n, 0, 0])

    # Compute the u and v vectors
    det_u = np.array([0, u_sign * 1, 0])
    det_v = np.array([0, 0, v_sign * 1])

    det_u = rot_p.apply(det_u)
    det_u = rot_t.apply(det_u)
    det_u = rot_n.apply(det_u)

    det_v = rot_p.apply(det_v)
    det_v = rot_t.apply(det_v)
    det_v = rot_n.apply(det_v)

    det_normal = np.cross(det_u, det_v)

    # Compute (u0, v0) as the projection of the source on the detector
    vec_source_to_detector = det_center - src_pos
    t = np.dot(det_normal, vec_source_to_detector) / np.dot(det_normal, det_normal)
    projection_source = src_pos + t * vec_source_to_detector
    vec_detector_to_projection = projection_source - det_center
    u0 = np.dot(vec_detector_to_projection, det_u) / pu
    v0 = np.dot(vec_detector_to_projection, det_v) / pv

    return DetectorGeometry(
        det_u_xyz=det_u,
        det_v_xyz=det_v,
        det_center_xyz=det_center,
        src_pos_xyz=src_pos,
        u0=u0,
        v0=v0,
        pu=pu,
        pv=pv,
    )


def project_to_uv(points_xyz: NDArray, geo: DetectorGeometry) -> NDArray:
    """
    Project a list of XYZ points to UV coordinates using precomputed geometry.

    Parameters:
    - xyz_points: Array of (x, y, z) points in the sample volume (XYZ coordinates).
    - geo: DetectorGeometry dataclass with precomputed properties.

    Returns:
    - Array of (u, v) coordinates for each input point.
    """
    points_xyz = np.array(points_xyz, dtype=np.float32)

    det_normal_xyz = np.cross(geo.det_u_xyz, geo.det_v_xyz)

    vecs_src_2_pnts = points_xyz - geo.src_pos_xyz
    denoms = np.dot(det_normal_xyz, vecs_src_2_pnts.T)

    t0 = np.dot(det_normal_xyz, geo.det_center_xyz - geo.src_pos_xyz)

    # Project each rotated point onto the tilted detector
    uv_coords = np.zeros((len(points_xyz), 2), dtype=np.float32)
    for ii_pnt, (vec_src_2_pnt, denom) in enumerate(zip(vecs_src_2_pnts, denoms)):

        if np.isclose(denom, 0.0):
            uv_coords[ii_pnt] = (np.nan, np.nan)
            continue

        intersection = geo.src_pos_xyz + t0 / denom * vec_src_2_pnt
        vec_detector_to_intersection = intersection - geo.det_center_xyz

        # Project onto U and V axes
        u = np.dot(vec_detector_to_intersection, geo.det_u_xyz) / geo.pu + geo.u0
        v = np.dot(vec_detector_to_intersection, geo.det_v_xyz) / geo.pv + geo.v0

        uv_coords[ii_pnt] = (u, v)

    return uv_coords


def compute_xyz_rotated(r: float, z: float, ws: NDArray, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> NDArray:
    """
    Compute XYZ coordinates for an array of (r, z, w) inputs in the rotated volume.

    Parameters:
    - r: Radial distance from volume center.
    - z: Elevation (Z coordinates).
    - w: Array of rotation angles (omega) in radians.
    - sx, sy, sz: Voxel sizes in X, Y, Z directions.

    Returns:
    - Array of (x, y, z) coordinates in the rotated volume.
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
    return rot_xyz * np.array((sx, sy, sz))


def plot_projection_geometry(
    geo: DetectorGeometry,
    det_size_uv: tuple[int, int],
    vol_size_xyz: tuple[int, int, int],
    voxel_size_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Plot the projection geometry using matplotlib.

    Parameters:
    - geo: DetectorGeometry dataclass with precomputed properties.
    - det_size_uv: Detector size in number of pixels in U and V directions.
    - vol_size_xyz: Volume size in number of voxels in X, Y, and Z directions.
    - voxel_size_xyz: Voxel sizes in X, Y, Z directions.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the source
    ax.scatter(*geo.src_pos_xyz, color='r', s=100, label='Source')
    # print(geo.src_pos_xyz)

    # Plot the detector
    det_corners = np.array(
        [
            geo.det_center_xyz + geo.det_u_xyz * det_size_uv[0] * geo.pu / 2 + geo.det_v_xyz * det_size_uv[1] * geo.pv / 2,
            geo.det_center_xyz + geo.det_u_xyz * det_size_uv[0] * geo.pu / 2 - geo.det_v_xyz * det_size_uv[1] * geo.pv / 2,
            geo.det_center_xyz - geo.det_u_xyz * det_size_uv[0] * geo.pu / 2 - geo.det_v_xyz * det_size_uv[1] * geo.pv / 2,
            geo.det_center_xyz - geo.det_u_xyz * det_size_uv[0] * geo.pu / 2 + geo.det_v_xyz * det_size_uv[1] * geo.pv / 2,
        ]
    )
    # print(det_corners)

    detector = Poly3DCollection([det_corners], alpha=0.5, linewidths=1, edgecolors='k')
    detector.set_facecolor('b')
    ax.add_collection3d(detector)

    # Plot the volume
    x, y, z = [size * voxel_size / 2 for size, voxel_size in zip(vol_size_xyz, voxel_size_xyz)]

    # Create a cube
    cube_vertices = np.array(
        [
            [-x, -y, -z],
            [x, -y, -z],
            [x, y, -z],
            [-x, y, -z],
            [-x, -y, z],
            [x, -y, z],
            [x, y, z],
            [-x, y, z],
        ]
    )
    # print(cube_vertices)

    # Create the 8 faces of the cube
    cube_faces = [
        [cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3]],  # Bottom face
        [cube_vertices[4], cube_vertices[5], cube_vertices[6], cube_vertices[7]],  # Top face
        [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]],  # Front face
        [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]],  # Back face
        [cube_vertices[1], cube_vertices[2], cube_vertices[6], cube_vertices[5]],  # Right face
        [cube_vertices[0], cube_vertices[3], cube_vertices[7], cube_vertices[4]],  # Left face
    ]

    volume = Poly3DCollection(cube_faces, alpha=0.1, linewidths=1, edgecolors='k')
    volume.set_facecolor('g')
    ax.add_collection3d(volume)

    # Plot vectors from the origin to the source and detector center
    ax.quiver(
        [0, 0],
        [0, 0],
        [0, 0],
        [geo.src_pos_xyz[0], geo.det_center_xyz[0]],
        [geo.src_pos_xyz[1], geo.det_center_xyz[1]],
        [geo.src_pos_xyz[2], geo.det_center_xyz[2]],
        color=['r', 'b'],
        arrow_length_ratio=0.1,
        label='Source and Detector Center',
    )

    # Plot vectors from the detector center to the u and v unit vectors
    ax.quiver(
        [geo.det_center_xyz[0], geo.det_center_xyz[0]],
        [geo.det_center_xyz[1], geo.det_center_xyz[1]],
        [geo.det_center_xyz[2], geo.det_center_xyz[2]],
        [geo.det_u_xyz[0] * det_size_uv[0] * geo.pu / 4, geo.det_v_xyz[0] * det_size_uv[1] * geo.pv / 4],
        [geo.det_u_xyz[1] * det_size_uv[0] * geo.pu / 4, geo.det_v_xyz[1] * det_size_uv[1] * geo.pv / 4],
        [geo.det_u_xyz[2] * det_size_uv[0] * geo.pu / 4, geo.det_v_xyz[2] * det_size_uv[1] * geo.pv / 4],
        color=['m', 'c'],
        arrow_length_ratio=0.1,
        label='U and V Vectors',
    )

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_aspect("equal")

    plt.show()


def test_cone_beam_ellipse():
    debug = False

    R = 30.0
    D = 20.0
    det_size_vu = 12

    geom = compute_detector_geometry(R=R, d=[D, 0.0, 0.0], p=0.0, t=0.0, n=0.0, pu=1.0, pv=1.0)
    if debug:
        print(f"{geom = }")
        vol_size_xyz = [det_size_vu - 2] * 3
        plot_projection_geometry(geom, det_size_uv=(det_size_vu,) * 2, vol_size_xyz=tuple(vol_size_xyz))

    r = det_size_vu / 2 - 1
    z_u = 2.5
    z_l = -1.5
    ws = np.deg2rad(np.arange(0, 360, 6))
    points_xyz_u = compute_xyz_rotated(r=r, z=z_u, ws=ws)
    points_xyz_l = compute_xyz_rotated(r=r, z=z_l, ws=ws)

    # fig, axs = plt.subplots(1, 1)
    # axs.plot(points_xyz[:, 0], points_xyz[:, 1])
    # axs.grid()
    # fig.tight_layout()
    # plt.show()

    prj_uv_coords_u = project_to_uv(points_xyz_u, geom)
    prj_uv_coords_l = project_to_uv(points_xyz_l, geom)

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
        D + R, geom_fit.acq_geom.D_pix
    ), f"The fitted source-detector distance is wrong ({geom_fit.acq_geom.D_pix:.6} vs expected: {D+R:.6})"
    assert np.isclose(
        R, geom_fit.acq_geom.R_pix
    ), f"The fitted source-sample distance is wrong ({geom_fit.acq_geom.R_pix:.6} vs expected: {R:.6})"


if __name__ == "__main__":
    # geometry = compute_detector_geometry(R=1e10, d=[1.0, 0.0, 0.0], p=0.0, t=0.0, n=0.0, pu=1.0, pv=1.0)
    geometry = compute_detector_geometry(R=160.0, d=[100.0, 0.0, 0.0], p=0.0, t=0.0, n=0.0, pu=1.0, pv=1.0)
    geometry = compute_detector_geometry(R=160.0, d=[100.0, 0.0, 0.0], p=0.0, t=0.0, n=np.pi / 4, pu=1.0, pv=1.0)
    # geometry = compute_detector_geometry(R=10.0, d=[1.0, 0.0, 0.0], p=0.0, t=0.0, n=np.pi / 2, pu=1.0, pv=1.0)
    # geometry = compute_detector_geometry(R=10.0, d=[1.0, 0.0, 0.0], p=0.0, t=0.0, n=-np.pi, pu=1.0, pv=2.0)
    print(geometry)

    plot_projection_geometry(geometry, (100, 100), (80, 80, 80))

    # Project points
    xyz_points_tst = np.array([[0, 1, 2], [3, 4, 5]])
    prj_uv_coords = project_to_uv(xyz_points_tst, geometry)
    print(prj_uv_coords)

    # Compute rotated XYZ for arrays of (r, z, w)
    ws = np.array([0, np.pi / 4, np.pi / 2])
    xyz_rotated = compute_xyz_rotated(r=2.0, z=1.0, ws=ws)
    print(xyz_rotated)
