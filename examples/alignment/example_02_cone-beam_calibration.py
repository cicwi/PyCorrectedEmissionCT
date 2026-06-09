"""
Example demonstrating the use of cone-beam geometry calibration routines.

@author: Nicola VIGANÒ, CEA-IRIG, Grenoble, France
"""

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

import corrct as cct


def _get_data(fid: h5py.File, data_path: str) -> NDArray:
    dataset = fid[data_path]
    if isinstance(dataset, h5py.Dataset):
        return np.array(dataset[()])
    else:
        raise ValueError(f"Path: {data_path}, is not a h5py.Dataset, but a {dataset} instead")


def _load_data(fname: str | Path) -> dict[str, NDArray]:
    with h5py.File(fname) as fid:
        return {k: _get_data(fid, f"/{k}") for k in fid.keys()}


data = _load_data("./data/calibration_scans.h5")

prj_size_vu = (data["scan_1"].shape[0], data["scan_1"].shape[2])

probe = cct.alignment.markers.create_marker_disk(prj_size_vu, 3.5)
pos_1, pos_2 = (cct.alignment.markers.track_marker(imgs, probe) for imgs in (data["scan_1"], data["scan_2"]))

pixel_size_um = float(data["pixel_size_um"])
orbit_radius_pix = float(data["orbit_radius_um"]) / pixel_size_um

fit_cb_geom = cct.alignment.cone_beam.FitConeBeamGeometry(
    prj_size_vu, points_ell1=pos_1, points_ell2=pos_2, pix_size_um=pixel_size_um, plot_result=True
)
acq_geom = fit_cb_geom.fit(r=orbit_radius_pix)

print(acq_geom)
cct.models.plot_projection_geometry(acq_geom.get_prj_geom(), acq_geom.get_vol_geom())

imgs_t = (data["scan_1"] + data["scan_2"]).astype(np.float32)

acq_geom = cct.alignment.cone_beam.tune_acquisition_geometry(
    acq_geom,
    data=imgs_t,
    angles_rot_rad=np.deg2rad(data["angles_deg"]),
    params=dict(
        D_pix=np.linspace(-24 * 2, 0, 9),
        theta_deg=np.linspace(-1, 1, 5),
        eta_deg=np.linspace(-1, 1, 5),
        phi_deg=np.linspace(-0.25, 0.25, 5),
        u0_pix=np.linspace(-1, 1, 9),
        v0_pix=np.linspace(-1, 1, 9),
    ),
    verbose=True,
)

print(acq_geom)
