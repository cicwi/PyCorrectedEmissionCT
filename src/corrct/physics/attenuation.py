"""
Incident beam and emidded radiation attenuation support.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import concurrent.futures as cf
import multiprocessing as mp
from collections.abc import Sequence
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from numpy.typing import ArrayLike, DTypeLike, NDArray
from tqdm.auto import tqdm

from corrct import _projector_backends as prj_backends
from corrct import models
from corrct.physics.xraylib_helper import get_compound, get_compound_cross_section

num_threads = round(np.log2(mp.cpu_count() + 1))

NDArrayFloat = NDArray[np.floating]
NDArrayInt = NDArray[np.integer]

CONVERT_UM_TO_CM = 1e-4


class AttenuationVolume:
    """Attenuation volume computation class."""

    incident_local: Union[NDArrayFloat, None]
    emitted_local: Union[NDArrayFloat, None]
    angles_rot_rad: NDArrayFloat
    angles_det_rad: NDArrayFloat

    dtype: DTypeLike

    vol_shape_zyx: NDArray
    maps: NDArray

    def __init__(
        self,
        incident_local: Union[NDArrayFloat, None],
        emitted_local: Union[NDArrayFloat, None],
        angles_rot_rad: ArrayLike,
        angles_det_rad: Union[NDArrayFloat, ArrayLike, float] = np.pi / 2,
        dtype: DTypeLike = np.float32,
    ):
        """
        Initialize the AttenuationVolume class.

        Raises
        ------
        ValueError
            In case no volumes were passed, or if they differed in shape.
        """
        self.incident_local = incident_local
        self.emitted_local = emitted_local
        self.angles_rot_rad = np.array(angles_rot_rad, ndmin=1)
        self.angles_det_rad = np.array(angles_det_rad, ndmin=1)

        self.dtype = dtype

        if self.incident_local is not None:
            self.vol_shape_zyx = np.array(self.incident_local.shape)

            if self.emitted_local is not None and np.any(self.vol_shape_zyx != self.emitted_local.shape):
                raise ValueError(
                    f"Incident volume shape ({self.incident_local.shape}) does not"
                    + f" match the emitted volume shape ({self.emitted_local.shape})"
                )
        elif self.emitted_local is not None:
            self.vol_shape_zyx = np.array(self.emitted_local.shape)
        else:
            raise ValueError("No attenuation volumes were given.")

        self.vol_shape_zyx = np.array(self.vol_shape_zyx, ndmin=1)

        num_dims = len(self.vol_shape_zyx)
        if num_dims not in [2, 3]:
            raise ValueError(f"Maps can only be 2D or 3D Arrays. A {num_dims}-dimensional was passed ({self.vol_shape_zyx}).")

    def _compute_attenuation_angle_in(self, local_att: NDArrayFloat, angle_rad: float) -> NDArray:
        return prj_backends.compute_attenuation(local_att, angle_rad, invert=False)[None, ...]

    def _compute_attenuation_angle_out(self, local_att: NDArrayFloat, angle_rad: float) -> NDArray:
        angle_det = angle_rad + self.angles_det_rad
        atts = np.empty(self.maps.shape[1:], dtype=self.dtype)
        for ii, a in enumerate(angle_det):
            atts[ii, ...] = prj_backends.compute_attenuation(local_att, a, invert=True)
        return atts

    def compute_maps(self, use_multithreading: bool = True, verbose: bool = True) -> None:
        """
        Compute the correction maps for each angle.

        Parameters
        ----------
        use_multithreading : bool, optional
            Use multi-threading for computing the attenuation maps. The default is True.
        verbose : bool, optional
            Show verbose output. The default is True.
        """
        num_rot_angles = len(self.angles_rot_rad)
        self.maps = np.ones([num_rot_angles, len(self.angles_det_rad), *self.vol_shape_zyx], dtype=self.dtype)

        def process_angles(
            func: Callable[[NDArray, float], NDArray], att_vol: NDArrayFloat, angles: NDArrayFloat, description: str
        ) -> None:
            if use_multithreading:
                with cf.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures_to_angle = {executor.submit(func, att_vol, a): (ii, a) for ii, a in enumerate(angles)}
                    try:
                        for f in tqdm(
                            cf.as_completed(futures_to_angle),
                            desc=description,
                            disable=(not verbose),
                            total=num_rot_angles,
                        ):
                            ii, a = futures_to_angle[f]
                            try:
                                self.maps[ii, ...] *= f.result()
                            except ValueError as exc:
                                raise RuntimeError(f"Angle {a} (#{ii}) generated an exception") from exc
                    except:
                        print("Shutting down..", end="", flush=True)
                        executor.shutdown(cancel_futures=True)
                        print("\b\b: Done.")
                        raise
            else:
                for ii, a in enumerate(tqdm(angles, desc=description, disable=(not verbose))):
                    self.maps[ii, ...] *= func(att_vol, a)

        if self.incident_local is not None:
            description = "Computing attenuation maps for incident beam"
            process_angles(
                self._compute_attenuation_angle_in, self.incident_local, angles=self.angles_rot_rad, description=description
            )

        if self.emitted_local is not None:
            description = "Computing attenuation maps for emitted photons"
            process_angles(
                self._compute_attenuation_angle_out, self.emitted_local, angles=self.angles_rot_rad, description=description
            )

    def plot_map(
        self,
        ax: Axes,
        rot_ind: int,
        det_ind: int = 0,
        slice_ind: Optional[int] = None,
        axes: Union[Sequence[int], NDArrayInt] = (-2, -1),
    ) -> Sequence[float]:
        """
        Plot the requested attenuation map.

        Parameters
        ----------
        ax : matplotlib axes
            The axes where to plot.
        rot_ind : int
            Rotation angle index.
        det_ind : int, optional
            Detector angle index. The default is 0.
        slice_ind : Optional[int], optional
            Volume slice index (for 3D volumes). The default is None.
        axes : Sequence[int] | NDArray, optional
            Axes of the slice. The default is (-2, -1).

        Returns
        -------
        Sequence[float]
            The extent of the axes plot (min-max coords).

        Raises
        ------
        ValueError
            In case a slice index is not passed for a 3D volume.
        """
        att_map = np.squeeze(self.get_maps(rot_ind=rot_ind, det_ind=det_ind))
        other_dim = np.squeeze(np.delete(np.arange(-3, 0), axes))
        if len(att_map.shape) == 3:
            if slice_ind is None:
                raise ValueError("Slice index is needed for 3D volumes. None was passed.")

            att_map = np.take(att_map, slice_ind, axis=int(other_dim))

        slice_shape = self.vol_shape_zyx[list(axes)]
        coords = [(-(s - 1) / 2, (s - 1) / 2) for s in slice_shape]

        extent = tuple(np.concatenate(coords))
        ax.imshow(att_map, extent=extent)

        if other_dim == -3:
            arrow_length = np.linalg.norm(slice_shape) / np.pi
            arrow_args = dict(
                width=arrow_length / 25,
                head_width=arrow_length / 8,
                head_length=arrow_length / 6,
                length_includes_head=True,
            )

            prj_geom = models.ProjectionGeometry.get_default_parallel()
            beam_i_geom = prj_geom.rotate(-self.angles_rot_rad[rot_ind])
            beam_e_geom = prj_geom.rotate(-(self.angles_rot_rad[rot_ind] + self.angles_det_rad[det_ind]))

            beam_i_dir = beam_i_geom.src_pos_xyz[0, :2] * arrow_length
            beam_i_orig = -beam_i_dir
            beam_e_dir = beam_e_geom.src_pos_xyz[0, :2] * arrow_length
            beam_e_orig = np.array([0, 0])

            ax.arrow(*beam_i_orig, *beam_i_dir, **arrow_args, fc="r", ec="r")
            ax.arrow(*beam_e_orig, *beam_e_dir, **arrow_args, fc="k", ec="k")

        return extent

    def get_maps(
        self,
        roi: Optional[ArrayLike] = None,
        rot_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
        det_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
    ) -> NDArray:
        """
        Return the attenuation maps.

        Parameters
        ----------
        roi : ArrayLike, optional
            The region-of-interest to select. The default is None.
        rot_ind : int, optional
            A specific rotation index, if only one is to be desired. The default is None.
        det_ind : int, optional
            A specific detector index, if only one is to be desired. The default is None.

        Returns
        -------
        NDArray
            The attenuation maps.
        """
        maps = self.maps

        if rot_ind is not None:
            if isinstance(rot_ind, int):
                rot_ind = slice(rot_ind, rot_ind + 1, 1)
            maps = maps[rot_ind, ...]

        if det_ind is not None:
            if isinstance(det_ind, int):
                det_ind = slice(det_ind, det_ind + 1, 1)
            maps = maps[:, det_ind, ...]

        if roi is not None:
            raise NotImplementedError("Extracting a region of interest is not supported, yet.")

        return maps

    def get_projector_args(
        self,
        roi: Optional[ArrayLike] = None,
        rot_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
        det_ind: Union[int, slice, Sequence[int], NDArrayInt, None] = None,
    ) -> dict[str, NDArray]:
        """
        Return the projector arguments.

        Parameters
        ----------
        roi : ArrayLike, optional
            The region-of-interest to select. The default is None.
        rot_ind : int, optional
            A specific rotation index, if only one is to be desired. The default is None.
        det_ind : int, optional
            A specific detector index, if only one is to be desired. The default is None.

        Returns
        -------
        dict[str, NDArray]
            A dictionary containing the attenuation maps and the detector angle.
        """
        if det_ind is None:
            det_angles = self.angles_det_rad
        else:
            det_angles = self.angles_det_rad[det_ind]
        return dict(att_maps=self.get_maps(roi=roi, rot_ind=rot_ind, det_ind=det_ind), angles_detectors_rad=det_angles)


def get_linear_attenuation_coefficient(
    compound: Union[str, dict], energy_keV: float, pixel_size_um: float, density: Union[float, None] = None
) -> float:
    """Compute the linear attenuation coefficient for given compound, energy, and pixel size.

    Parameters
    ----------
    compound : Union[str, dict]
        The compound for which we compute the linear attenuation coefficient
    energy_keV : float
        The energy of the photons
    pixel_size_um : float
        The pixel size in microns
    density : Union[float, None], optional
        The density of the compound (if different from the default value), by default None

    Returns
    -------
    float
        The linear attenuation coefficient
    """
    if isinstance(compound, str):
        compound = get_compound(compound)

    if density is not None:
        compound["density"] = density

    cmp_cs = get_compound_cross_section(compound, energy_keV)
    return pixel_size_um * CONVERT_UM_TO_CM * compound["density"] * cmp_cs


def plot_emission_line_attenuation(
    compound: Union[str, dict],
    thickness_um: float,
    mean_energy_keV: float,
    fwhm_keV: float,
    line_shape: str = "lorentzian",
    num_points: int = 201,
    plot_lines_mean: bool = True,
) -> None:
    """Plot spectral attenuation of a given line.

    Parameters
    ----------
    compound : Union[str, dict]
        Compound to consider
    thickness_um : float
        Thickness of the compound (in microns)
    mean_energy_keV : float
        Average energy of the line
    fwhm_keV : float
        Full-width half-maximum of the line
    line_shape : str, optional
        Shape of the line, by default "lorentzian".
        Options are: "gaussian" | "lorentzian" | "sech**2".
    num_points : int, optional
        number of discretization points, by default 201
    plot_lines_mean : bool, optional
        Whether to plot the line mean and the effective line mean, by default True

    Raises
    ------
    ValueError
        When an unsupported line is chosen.
    """
    xc = np.linspace(-0.5, 0.5, num_points)

    if line_shape.lower() == "gaussian":
        xc *= fwhm_keV * 3
        yg = np.exp(-4 * np.log(2) * (xc**2) / (fwhm_keV**2))
    elif line_shape.lower() == "lorentzian":
        xc *= fwhm_keV * 13
        hwhm_keV = fwhm_keV / 2
        yg = hwhm_keV / (xc**2 + hwhm_keV**2)
    elif line_shape.lower() == "sech**2":
        # doi: 10.1364/ol.20.001160
        xc *= fwhm_keV * 4
        tau = fwhm_keV / (2 * np.arccosh(np.sqrt(2)))
        yg = 1 / np.cosh(xc / tau) ** 2
    else:
        raise ValueError(f"Unknown beam shape: {line_shape.lower()}")

    nrgs_keV = xc + mean_energy_keV

    if isinstance(compound, str):
        compound = get_compound(compound)

    atts = np.empty_like(yg)

    for ii, nrg in enumerate(nrgs_keV):
        cmp_cs = get_compound_cross_section(compound, nrg)
        atts[ii] = np.exp(-thickness_um * CONVERT_UM_TO_CM * compound["density"] * cmp_cs)

    yg = yg / np.max(yg)
    mean_effective_energy_keV = float(nrgs_keV.dot(yg * atts) / yg.dot(atts))

    fig, axs_line = plt.subplots(1, 1)
    pl_line = axs_line.plot(nrgs_keV, yg, label="$I_0$", color="C0")
    if plot_lines_mean:
        axs_line.axvline(mean_energy_keV, linestyle="--", color="C0")
    axs_line.tick_params(axis="y", labelcolor="C0", labelsize=14)
    axs_line.tick_params(axis="x", labelsize=14)
    axs_line.set_ylabel("Intensity", color="C0", fontsize=14)
    axs_atts = axs_line.twinx()
    pl_atts = axs_atts.plot(nrgs_keV, atts, label="exp(-$\\mu (E) x)$", color="C1")
    pl_line_att = axs_atts.plot(nrgs_keV, yg * atts, label=r"$I_{measured}$", color="C2")
    if plot_lines_mean:
        axs_atts.axvline(mean_effective_energy_keV, linestyle="--", color="C2")
    axs_atts.tick_params(axis="y", labelcolor="C1", labelsize=14)
    axs_atts.set_ylabel("Transmittance", color="C1", fontsize=14)
    all_pls = pl_line + pl_atts + pl_line_att
    axs_atts.legend(all_pls, [str(pl.get_label()) for pl in all_pls], fontsize=13)
    axs_line.grid()
    fig.tight_layout()

    I_lin = np.sum(yg * atts[num_points // 2])
    I_meas = yg.dot(atts)
    print(f"Expected intensity: {I_lin}, measured: {I_meas} ({I_meas / I_lin:%})")
    print(f"Expected mean energy: {nrgs_keV.dot(yg / np.sum(yg)):.5}, effective: {mean_effective_energy_keV:.5}")


def plot_transmittance_decay(
    compounds: Union[str, dict, Sequence[Union[str, dict]]],
    mean_energy_keV: float,
    thickness_range_um: tuple[float, float, int] = (0.0, 10.0, 101),
) -> None:
    """Plot transmittance decay curve(s) for the given compound(s) at a given energy and thickness range.

    Parameters
    ----------
    compounds : str | dict | Sequence[str | dict]
        The compound(s) description
    mean_energy_keV : float
        The mean photon energy
    thickness_range_um : tuple[float, float, int], optional
        The thickness range as (start, end, num_points), by default (0.0, 10.0, 101)
    """
    if isinstance(compounds, (str, dict)):
        compounds = [compounds]

    compounds = [get_compound(c) if isinstance(c, str) else c for c in compounds]

    thicknesses_um = np.linspace(*thickness_range_um)

    atts = np.zeros((len(compounds), len(thicknesses_um)), dtype=np.float32)
    for ii, cmp in enumerate(compounds):
        cmp_cs = get_compound_cross_section(cmp, mean_energy_keV)
        atts[ii] = np.exp(-thicknesses_um * CONVERT_UM_TO_CM * cmp["density"] * cmp_cs)

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    for ii, cmp in enumerate(compounds):
        axs.plot(thicknesses_um, atts[ii], label=cmp["name"])
    axs.legend(fontsize=13)
    axs.grid()
    axs.tick_params(labelsize=14)
    axs.set_xlabel("Thickness [$\mu m$]", fontsize=14)
    axs.set_xlim(thickness_range_um[0], thickness_range_um[1])
    axs.set_ylabel("Transmittance", fontsize=14)
    axs.set_ylim(0.0, 1.0)
    axs.set_title(f"Transmittance curve at {mean_energy_keV:.2f} keV", fontsize=14)
    fig.tight_layout()
    plt.plot(block=False)
