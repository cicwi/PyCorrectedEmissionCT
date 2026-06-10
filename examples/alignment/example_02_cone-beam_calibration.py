"""
Example demonstrating the use of cone-beam geometry calibration routines.

@author: Nicola VIGANÒ, CEA-IRIG, Grenoble, France
"""

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from corrct.alignment.cone_beam import FitConeBeamGeometry, tune_acquisition_geometry
from corrct.alignment.markers import create_marker_disk, track_marker
from corrct.models import plot_projection_geometry


def _get_data(fid: h5py.File, data_path: str) -> NDArray:
    dataset = fid[data_path]
    if isinstance(dataset, h5py.Dataset):
        return np.array(dataset[()])
    else:
        raise ValueError(f"Path: {data_path}, is not a h5py.Dataset, but a {dataset} instead")


def _load_data(fname: str | Path) -> dict[str, NDArray]:
    with h5py.File(fname) as fid:
        return {k: _get_data(fid, f"/{k}") for k in fid.keys()}


try:
    data = _load_data("./data/calibration_scans.h5")
except FileNotFoundError as exc:
    raise ValueError("Please download the example dataset from https://doi.org/10.5281/zenodo.20559974") from exc

prj_size_vu = (data["scan_1"].shape[0], data["scan_1"].shape[2])

probe = create_marker_disk(prj_size_vu, 3.5)
pos_l, pos_u = (track_marker(imgs, probe) for imgs in (data["scan_1"], data["scan_2"]))

pixel_size_um = float(data["pixel_size_um"])
orbit_radius_pix = float(data["orbit_radius_um"]) / pixel_size_um

fit_cb_geom = FitConeBeamGeometry(
    prj_size_vu, points_ell1=pos_u, points_ell2=pos_l, pix_size_um=pixel_size_um, plot_result=True
)
acq_geom = fit_cb_geom.fit(r=orbit_radius_pix)

print(acq_geom)

imgs_t = (data["scan_1"] + data["scan_2"]).astype(np.float32)
angles_rot_rad = np.deg2rad(data["angles_deg"])

acq_geom = tune_acquisition_geometry(
    acq_geom,
    data=imgs_t,
    angles_rot_rad=angles_rot_rad,
    params=dict(
        theta_deg=np.linspace(-1, 1, 5),
        phi_deg=np.linspace(-0.25, 0.25, 5),
        u0_pix=np.linspace(-1, 1, 9),
        v0_pix=np.linspace(-1, 1, 9),
    ),
    verbose=True,
)

print(acq_geom)
plot_projection_geometry(acq_geom.get_prj_geom(), acq_geom.get_vol_geom())
