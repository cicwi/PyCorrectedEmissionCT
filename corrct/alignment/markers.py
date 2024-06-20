#!/usr/bin/env python3
"""
Fiducial marker tracking routines.

@author: Nicola VIGANÒ, ESRF - The European Synchrotron, Grenoble, France,
and CEA-IRIG, Grenoble, France
"""

from typing import Union

import numpy as np
import scipy.ndimage as spimg
from numpy.typing import ArrayLike, NDArray

import matplotlib.pyplot as plt

from . import fitting


def cm2inch(dims: Union[ArrayLike, NDArray]) -> tuple[float]:
    """Convert cm into inch.

    Parameters
    ----------
    dims : Union[ArrayLike, NDArray]
        The dimentions of the object in cm

    Returns
    -------
    tuple[float]
        The output dimensions in inch
    """
    return tuple(np.array(dims) / 2.54)


def track_marker(prj_data: NDArray, marker_vu: NDArray, stack_axis: int = -2) -> NDArray:
    """Track marker position in a stack of images.

    Parameters
    ----------
    prj_data_vwu : NDArray
        The projection data.
    marker_vu : NDArray
        The fiducial marker to track in VU.
    stack_axis : int, optional
        The axis along which the images are stacked. The default is -2.

    Returns
    -------
    NDArray
        List of positions for each image.
    """
    marker_v1u = np.expand_dims(marker_vu, stack_axis).astype(np.float32)
    marker_pos = fitting.fit_shifts_vu_xc(prj_data, marker_v1u, stack_axis=stack_axis, normalize_fourier=False)
    return marker_pos + np.array(marker_vu.shape)[:, None] / 2


def create_marker_disk(
    data_shape_vu: Union[ArrayLike, NDArray], radius: float, super_sampling: int = 5, conv: bool = True
) -> NDArray:
    """
    Create a Disk probe object, that will be used for tracking a calibration object's movement.

    Parameters
    ----------
    data_shape_vu : ArrayLike
        Shape of the images (vertical, horizontal).
    radius : float
        Radius of the probe.
    super_sampling : int, optional
        Super sampling of the coordinates used for creation. The default is 5.
    conv : bool, optional
        Whether to convolve the initial probe with itself. The default is True.

    Returns
    -------
    NDArray
        An image of the same size as the projections, that contains the marker in the center.
    """
    data_shape_vu = np.array(data_shape_vu, dtype=int) * super_sampling

    # coords = [np.linspace(-(s - 1) / 2, (s - 1) / 2, s, dtype=np.float32) for s in data_shape_vu]
    coords = [np.fft.fftfreq(d, 1 / d) for d in data_shape_vu]
    coords = np.stack(np.meshgrid(*coords, indexing="ij"), axis=0)
    pix_rr = np.sqrt(np.sum(coords**2, axis=0))

    probe = pix_rr < radius * super_sampling
    probe = np.roll(probe, super_sampling // 2, axis=tuple(np.arange(len(data_shape_vu))))
    new_shape = np.stack([data_shape_vu // super_sampling, np.ones_like(data_shape_vu) * super_sampling], axis=1).flatten()
    probe = probe.reshape(new_shape)
    probe = np.mean(probe, axis=tuple(np.arange(1, len(data_shape_vu) * 2, 2, dtype=int)))

    probe = np.fft.fftshift(probe)

    if conv:
        probe = spimg.convolve(probe, probe)

    return probe


class MarkerTrackingVisualizer:
    """Plotting class to assess the marker tracking quality."""

    def __init__(
        self,
        fitted_positions_vu: Union[ArrayLike, NDArray],
        images: NDArray,
        marker: NDArray,
        trajectory: Union[fitting.Trajectory, None] = None,
    ) -> None:
        """Initialize the visualization utility for checking the marker position fitting.

        Parameters
        ----------
        fitted_positions_vu : Union[ArrayLike, NDArray]
            The fitted positions of the marker
        imgs : NDArray
            The original images
        disk : NDArray
            The marker image
        trajectory : Union[fitting.Trajectory, None], optional
            The trajectory object that the points are supposed to follow, by default None
        """
        self.positions_vu = np.array(fitted_positions_vu)
        self.images = images
        self.marker = marker
        self.global_lims = False

        self.trajectory = trajectory

        self.curr_pos = 0

        if self.trajectory is not None:
            uus = np.sort(self.positions_vu[1, :])
            self.vvs = self.trajectory(uus)

        self.fig, self.axs = plt.subplots(1, 3, figsize=cm2inch([36, 12]))  # , sharex=True, sharey=True
        self.axs[2].imshow(self.marker)
        self.axs[0].set_xlim(0, self.images.shape[-1])
        self.axs[0].set_ylim(self.images.shape[-3], 0)
        self.fig.tight_layout()
        self._update()

        self.fig.canvas.mpl_connect("key_press_event", self._key_event)
        self.fig.canvas.mpl_connect("scroll_event", self._scroll_event)

    def _update(self) -> None:
        self.curr_pos = self.curr_pos % self.images.shape[-2]

        for img in self.axs[0].get_images():
            img.remove()
        x_lims = self.axs[0].get_xlim()
        y_lims = self.axs[0].get_ylim()
        self.axs[0].cla()
        self.axs[0].set_xlim(x_lims[0], x_lims[1])
        self.axs[0].set_ylim(y_lims[0], y_lims[1])

        for img in self.axs[1].get_images():
            img.remove()
        self.axs[1].cla()

        self.axs[0].plot(self.positions_vu[1, :], self.positions_vu[0, :], "bo-", markersize=4)
        self.axs[0].scatter(self.positions_vu[1, self.curr_pos], self.positions_vu[0, self.curr_pos], c="r")

        if self.trajectory is not None:
            uus = np.sort(self.positions_vu[1, :])
            for vvs in self.vvs:
                self.axs[0].plot(uus, vvs, "g")
        self.axs[0].grid()

        if self.global_lims:
            vmin = self.images.min()
            vmax = self.images.max()
        else:
            vmin = self.images[:, self.curr_pos, :].min()
            vmax = self.images[:, self.curr_pos, :].max()

        img = self.axs[1].imshow(self.images[:, self.curr_pos, :], vmin=vmin, vmax=vmax)
        self.axs[1].scatter(self.positions_vu[1, self.curr_pos], self.positions_vu[0, self.curr_pos], c="r")
        self.axs[1].set_title(f"Range: [{vmin}, {vmax}]")
        # plt.colorbar(im, ax=self.axs[1])
        self.fig.canvas.draw()

    def _key_event(self, evnt) -> None:
        if evnt.key == "right":
            self.curr_pos += 1
        elif evnt.key == "left":
            self.curr_pos -= 1
        elif evnt.key == "up":
            self.curr_pos += 1
        elif evnt.key == "down":
            self.curr_pos -= 1
        elif evnt.key == "pageup":
            self.curr_pos += 10
        elif evnt.key == "pagedown":
            self.curr_pos -= 10
        elif evnt.key == "escape":
            plt.close(self.fig)
        elif evnt.key == "ctrl+l":
            self.global_lims = not self.global_lims
        else:
            print(evnt.key)
            return

        self._update()

    def _scroll_event(self, evnt) -> None:
        if evnt.button == "up":
            self.curr_pos += 1
        elif evnt.button == "down":
            self.curr_pos -= 1
        else:
            print(evnt.key)
            return

        self._update()
