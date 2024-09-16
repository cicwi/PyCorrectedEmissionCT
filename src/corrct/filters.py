"""
Filtered back-projection filters.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from scipy.interpolate import interp1d
from skimage.transform.radon_transform import _get_fourier_filter

from .operators import BaseTransform
from .processing import circular_mask


try:
    import pywt

    has_pywt = True
except ImportError:
    print("WARNING: You need to install PyWavelets to benefit from wavelet bases.")

    has_pywt = False


class BasisOptions(ABC, Mapping):
    """Options for the different types of bases."""

    def __len__(self) -> int:
        """Return the number of options.

        Returns
        -------
        int
            The number of options.
        """
        return self.__dict__.__len__()

    def __getitem__(self, k: Any) -> Any:
        """Return the selected option.

        Parameters
        ----------
        k : Any
            The key of the selected option.

        Returns
        -------
        Any
            The selected option.
        """
        return self.__dict__.__getitem__(k)

    def __iter__(self) -> Any:
        """Iterate the options list.

        Returns
        -------
        Any
            The following option.
        """
        return iter(self.__dict__)


@dataclass
class BasisOptionsBlocks(BasisOptions):
    """Options for the wavelet bases."""

    binning_start: Optional[int] = 2
    binning_type: str = "exponential"
    order: int = 1
    normalized: bool = True


@dataclass
class BasisOptionsWavelets(BasisOptions):
    """Options for the wavelet bases."""

    wavelet: str = "bior2.2"
    level: int = 5
    norm: float = 1.0


def create_basis(
    num_pixels: int,
    binning_start: Optional[int] = 2,
    binning_type: str = "exponential",
    normalized: bool = False,
    order: int = 1,
    dtype: DTypeLike = np.float32,
) -> NDArray:
    """Compute filter basis matrix.

    Parameters
    ----------
    num_pixels : int
        Number of filter fixels.
    binning_start : Optional[int], optional
        Starting displacement of the binning, by default 2.
    binning_type : str, optional
        Type of pixel binning, by default "exponential".
    normalized : bool, optional
        Whether to normalize the bins by the window size, by default True.
    order : int, optional
        Order of the basis functions. Only 0 and 1 supported, by default 1.
    dtype : DTypeLike, optional
        Data type, by default np.float32.

    Returns
    -------
    NDArray
        The filter basis.
    """
    filter_positions = np.abs(np.fft.fftfreq(num_pixels, 1 / num_pixels))

    window_size = 1
    window_position = 0

    basis_r = []
    while window_position < filter_positions.max():
        basis_tmp = np.zeros(filter_positions.shape, dtype=dtype)

        if order == 0:
            binning_positions = np.logical_and(
                window_position <= filter_positions, filter_positions < (window_position + window_size)
            )

            basis_val = 1.0
            if normalized:
                basis_val /= window_size

            basis_tmp[binning_positions] = basis_val
        else:
            basis_tmp = np.fmax(1.0 - filter_positions / (window_position + window_size), 0.0)

        basis_r.append(basis_tmp)
        window_position += window_size

        if binning_start is not None and window_position > binning_start:
            if binning_type == "exponential":
                window_size = 2 * window_size
            elif binning_type == "incremental":
                window_size += 1
            elif binning_type == "custom":
                window_size += int(np.sqrt(window_position - binning_start))
            else:
                raise ValueError(f"Invalid 'binning_type' = {binning_type}.")

    basis_r = np.array(basis_r, dtype=dtype)

    if order > 0:
        for ii in range(1, basis_r.shape[0]):
            for jj in range(0, ii):
                other_vec = basis_r[jj, ...]
                other_vec_norm_2 = other_vec.dot(other_vec)
                basis_r[ii, ...] -= other_vec * other_vec.dot(basis_r[ii, ...]) / other_vec_norm_2
        if normalized:
            basis_r /= np.linalg.norm(basis_r, ord=1, axis=-1, keepdims=True)

    return basis_r


def create_basis_wavelet(
    num_pixels: int,
    wavelet: str = "bior2.2",
    level: int = 5,
    norm: float = 1.0,
    dtype: DTypeLike = np.float32,
) -> NDArray:
    """Compute filter basis matrix.
    Parameters
    ----------
    num_pixels : int
        Number of wavelet filters.
    wavelet: str, optional
        The wavelet to use, by default "bior4.4".
    level : int, optional
        The decomposition level to reach, by default 5.
    norm : float, optional
        The norm to use, by default 1.0.
    dtype : DTypeLike, optional
        Data type, by default np.float32.
    Returns
    -------
    NDArray
        The filter basis.
    """
    if not has_pywt:
        print("WARNING: You need to install PyWavelets to benefit from wavelet bases.")
        raise ImportError("PyWavelets (pywt) module not found.")

    # max_level = pywt.swt_max_level(num_pixels)
    # if level > max_level:
    #     print(f"WARNING: Requested level {level} is too high for {num_pixels} pixels. Max allowed is {max_level}.")
    #     level = max_level

    w = pywt.Wavelet(wavelet)

    dec_lo = np.trim_zeros(w.dec_lo)
    dec_hi = np.trim_zeros(w.dec_hi)

    crop_size_l = num_pixels // 2
    crop_size_u = num_pixels - crop_size_l

    pad_size_u = (num_pixels - len(dec_hi)) // 2
    pad_size_l = num_pixels - len(dec_hi) - pad_size_u
    pad_width = (pad_size_l, pad_size_u)

    basis_hi_tmp = np.pad(dec_hi, pad_width=np.array(pad_width))

    pad_size_u = (num_pixels - len(dec_lo)) // 2
    pad_size_l = num_pixels - len(dec_lo) - pad_size_u
    pad_width = (pad_size_l, pad_size_u)

    basis_lo_tmp = np.pad(dec_lo, pad_width=np.array(pad_width))

    coords = np.fft.fftshift(np.fft.fftfreq(num_pixels, 1 / num_pixels))

    basis_r = []
    basis_r.append(basis_hi_tmp)
    for _ in range(level - 1):
        basis_lo_old = basis_lo_tmp.copy()

        int_obj = interp1d(coords, basis_hi_tmp, kind="linear")
        basis_hi_tmp = int_obj((coords + 1 - len(coords) % 2) / 2)
        basis_hi_tmp = np.convolve(basis_hi_tmp, basis_lo_old, mode="same")

        int_obj = interp1d(coords, basis_lo_tmp, kind="linear")
        basis_lo_tmp = int_obj((coords + 1 - len(coords) % 2) / 2)
        basis_lo_tmp = np.convolve(basis_lo_tmp, basis_lo_old, mode="same")

        basis_r.append(basis_hi_tmp)
    basis_r.append(basis_lo_tmp)

    basis_r = np.array(basis_r, dtype=dtype)
    basis_r /= np.linalg.norm(basis_r, axis=-1, ord=norm, keepdims=True)

    return np.fft.ifftshift(basis_r, axes=(-1,))


class Filter(ABC):
    """Base FBP filter."""

    fbp_filter: NDArray[np.floating]
    pad_mode: str
    use_rfft: bool
    dtype: DTypeLike

    def __init__(
        self,
        fbp_filter: Union[ArrayLike, NDArray[np.floating], None],
        pad_mode: str,
        use_rfft: bool,
        dtype: DTypeLike,
    ) -> None:
        """Initialize Base FBP filter.

        Parameters
        ----------
        fbp_filter : Union[ArrayLike, NDArray[np.floating], None]
            The filter.
        pad_mode : str
            The padding mode.
        use_rfft : bool
            Whethert to use the `rfft` or complex `fft`.
        dtype : DTypeLike
            The data type of the filter.
        """
        self.dtype = dtype
        self.pad_mode = pad_mode.lower()
        self.use_rfft = use_rfft

        if fbp_filter is None:
            self.fbp_filter = np.array([1.0], dtype=dtype)
        else:
            self.fbp_filter = np.array(np.real(fbp_filter), dtype=dtype)

    def get_padding_size(self, data_wu_shape: Sequence[int]) -> int:
        """Compute the projection padding size for a linear convolution.

        Parameters
        ----------
        data_wu_shape : Sequence[int]
            The shape of the data

        Returns
        -------
        int
            The padding size of the last dimension.
        """
        return max(64, int(2 ** np.ceil(np.log2(2 * data_wu_shape[-1]))))

    def to_fourier(self, data_wu: NDArray) -> NDArray:
        if self.use_rfft:
            return np.fft.rfft(data_wu, axis=-1)
        else:
            return np.fft.fft(data_wu, axis=-1)

    def to_real(self, data_wu: NDArray) -> NDArray:
        if self.use_rfft:
            return np.fft.irfft(data_wu, axis=-1)
        else:
            return np.fft.ifft(data_wu, axis=-1).real

    @property
    def filter_fourier(self) -> NDArray[np.floating]:
        """Fourier representation of the filter.

        Returns
        -------
        NDArray[np.floating]
            The filter in Fourier.
        """
        return self.fbp_filter.copy()

    @property
    def filter_real(self) -> NDArray[np.floating]:
        """Real-space representation of the filter.

        Returns
        -------
        NDArray[np.floating]
            The filter in real-space.
        """
        fbp_filter_r = self.to_real(self.fbp_filter)
        return np.fft.fftshift(fbp_filter_r, axes=(-1,))

    @filter_real.setter
    def filter_real(self, fbp_filter_r: NDArray[np.floating]) -> None:
        self.fbp_filter = self.to_fourier(fbp_filter_r)

    @filter_fourier.setter
    def filter_fourier(self, fbp_filter_f: NDArray[np.floating]) -> None:
        self.fbp_filter = fbp_filter_f.copy()

    @property
    def num_filters(self) -> int:
        return np.array(self.fbp_filter, ndmin=2).shape[-2]

    def apply_filter(self, data_wu: NDArray, fbp_filter: Optional[NDArray] = None) -> NDArray:
        """Apply the filter to the data_wu.

        Parameters
        ----------
        data_wu : NDArray
            The sinogram.
        fbp_filter : NDArray, optional
            The filter to use. The default is None

        Returns
        -------
        NDArray
            The filtered sinogram.
        """
        data_wu_shape = data_wu.shape
        if fbp_filter is None:
            local_filter = self.fbp_filter
        else:
            local_filter = fbp_filter

        prj_size_pad = self.get_padding_size(data_wu_shape)

        pad_edge_u = (prj_size_pad - data_wu_shape[-1]) / 2
        pad_width = np.zeros((len(data_wu_shape), 2), dtype=int)
        pad_width[-1, :] = (int(np.ceil(pad_edge_u)), int(np.floor(pad_edge_u)))

        prj_pad = np.pad(data_wu, pad_width=tuple(pad_width), mode=self.pad_mode)  # type: ignore
        prj_pad = np.roll(prj_pad, shift=-pad_width[-1][0], axis=-1)

        prj_f = self.to_fourier(prj_pad)

        local_filter = np.array(local_filter, ndmin=2)

        num_filters = local_filter.shape[-2]

        if num_filters > 1:
            prjs_f = [np.array([])] * num_filters

            for ii, f in enumerate(local_filter):
                prj_f_ii = prj_f * np.array(f, ndmin=len(data_wu_shape))
                prj_f_ii = self.to_real(prj_f_ii)
                prjs_f[ii] = prj_f_ii[..., : data_wu_shape[-1]]

            return np.array(prjs_f)
        else:
            prj_f *= np.array(local_filter, ndmin=len(data_wu_shape))
            return self.to_real(prj_f)[..., : data_wu_shape[-1]]

    @abstractmethod
    def compute_filter(self, data_wu: NDArray) -> None:
        """Compute the FBP filter for the given data.

        Parameters
        ----------
        data_wu : NDArray
            The reference sinogram / projection data.
        """

    def __call__(self, data_wu: NDArray) -> NDArray:
        """Filter the sinogram, by first computing the filter, and then applying it.

        Parameters
        ----------
        data_wu : NDArray
            The unfiltered sinogram.

        Returns
        -------
        NDArray
            The filtered sinogram.
        """
        self.compute_filter(data_wu)
        return self.apply_filter(data_wu, self.fbp_filter)

    def plot_filters(self, fourier_abs: bool = False):
        filters_r = np.array(self.filter_real, ndmin=2)
        filters_f = np.array(self.filter_fourier, ndmin=2)

        f, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ii in range(self.num_filters):
            axes[0].plot(filters_r[ii, ...], label=f"Filter-{ii}")
        axes[0].set_title("Real-space")
        axes[0].set_xlabel("Pixel")

        for ii in range(self.num_filters):
            filt_f = filters_f[ii, ...]
            if fourier_abs:
                filt_f = np.abs(filt_f)
            axes[1].plot(filt_f, label=f"Filter-{ii}")
        axes[1].set_title("Fourier-space")
        axes[1].set_xlabel("Frequency")

        axes[0].grid()
        axes[1].grid()
        if self.num_filters > 1:
            axes[1].legend()
        f.tight_layout()


class FilterCustom(Filter):
    """Custom FBP filter."""

    def __init__(
        self,
        fbp_filter: Union[ArrayLike, NDArray[np.floating], None],
        pad_mode: str = "constant",
        use_rfft: bool = True,
        dtype: DTypeLike = np.float32,
    ) -> None:
        """Initialize Custom FBP filter.

        Parameters
        ----------
        fbp_filter : Union[ArrayLike, NDArray[np.floating], None]
            The filter.
        pad_mode : str, optional
            The padding mode, by default "constant".
        use_rfft : bool, optional
            Whethert to use the `rfft` or complex `fft`, by default True.
        dtype : DTypeLike, optional
            The data type of the filter, by default np.float32.
        """
        super().__init__(fbp_filter, pad_mode, use_rfft, dtype)

    def compute_filter(self, data_wu: NDArray) -> None:
        """Provide dummy implementation, because it is not required."""


class FilterFBP(Filter):
    """Traditional FBP filter."""

    filter_name: str

    FILTERS = ("ramp", "shepp-logan", "cosine", "hamming", "hann")

    def __init__(
        self, filter_name: str = "ramp", pad_mode: str = "constant", use_rfft: bool = True, dtype: DTypeLike = np.float32
    ) -> None:
        """Initialize traditional FBP filter.

        Parameters
        ----------
        filter_name : str
            The filter name
        use_rfft : bool, optional
            Whethert to use the `rfft` or complex `fft`, by default True
        dtype : DTypeLike, optional
            The type of the filter, by default np.float32
        """
        if filter_name.lower() not in self.FILTERS:
            raise ValueError(f"Unknown filter {filter_name}. Available filters: {self.FILTERS}")

        super().__init__(fbp_filter=None, pad_mode=pad_mode, use_rfft=use_rfft, dtype=dtype)

        self.filter_name = filter_name.lower()

    def compute_filter(self, data_wu: NDArray) -> None:
        """Compute the traditional FBP filter for the given data.

        Parameters
        ----------
        data_wu : NDArray
            The reference sinogram / projection data.
        """
        prj_size_pad = self.get_padding_size(data_wu.shape)

        self.fbp_filter = _get_fourier_filter(prj_size_pad, self.filter_name)
        self.fbp_filter = np.squeeze(self.fbp_filter) * np.pi / (2 * data_wu.shape[-2])

        if self.use_rfft:
            self.fbp_filter = self.fbp_filter[: (self.fbp_filter.shape[-1]) // 2 + 1]

    def get_available_filters(self) -> Sequence[str]:
        """Provide available FBP filters.

        Returns
        -------
        Sequence[str]
            The available filters.
        """
        return self.FILTERS


class FilterMR(Filter):
    """Data dependent FBP filter.

    This is a simplified implementation from:

    [1] Pelt, D. M., & Batenburg, K. J. (2014). Improving filtered backprojection
    reconstruction by data-dependent filtering. Image Processing, IEEE
    Transactions on, 23(11), 4750-4762.

    Code inspired by: https://github.com/dmpelt/pymrfbp
    """

    projector: BaseTransform

    binning_type: str
    binning_start: Union[int, None]

    lambda_smooth: Union[float, None]

    is_initialized: bool

    def __init__(
        self,
        projector: BaseTransform,
        binning_type: str = "exponential",
        binning_start: Union[int, None] = 2,
        lambda_smooth: Optional[float] = None,
        pad_mode: str = "constant",
        use_rfft: bool = True,
        dtype: DTypeLike = np.float32,
    ) -> None:
        """Initialize data-driven FBP filter.

        Parameters
        ----------
        projector : BaseTransform
            The projector to use for handling the data.
        start_exp_binning : int, optional
            From which distance to start exponentional binning. The default is 2.
        lambda_smooth : float, optional
            Smoothing parameter. The default is None.
        dtype : DTypeLike, optional
            Filter data type. The default is np.float32.
        """
        super().__init__(fbp_filter=None, pad_mode=pad_mode, use_rfft=use_rfft, dtype=dtype)

        self.projector = projector

        self.binning_type = binning_type.lower()
        if self.binning_type not in ("exponential", "incremental"):
            raise ValueError("Binning type should be either 'exponential' or 'incremental'.")
        self.binning_start = binning_start
        self.lambda_smooth = lambda_smooth

        self.is_initialized = False

    def initialize(self, data_wu_shape: Sequence[int]) -> None:
        """Initialize filter.

        Parameters
        ----------
        data_wu_shape : Sequence[int]
            Shape of the data.
        """
        num_pad_pixels = self.get_padding_size(data_wu_shape)

        self.basis_r = create_basis(
            num_pad_pixels, binning_type=self.binning_type, binning_start=self.binning_start, dtype=self.dtype
        )

        self.basis_f = self.to_fourier(self.basis_r).real

        self.is_initialized = True

    def compute_filter(self, data_wu: NDArray) -> None:
        """Compute the filter.

        Parameters
        ----------
        data_wu : NDArray
            The sinogram.
        projector : ProjectorOperator
            The projector used in the FBP.
        """
        if not self.is_initialized:
            self.initialize(data_wu.shape)

        num_sino_pixels = data_wu.shape[-1]

        sino_size = data_wu.shape[-2] * num_sino_pixels
        nrows = sino_size
        ncols = self.basis_f.shape[-2]

        if self.lambda_smooth:
            grad_vol_size = num_sino_pixels * (num_sino_pixels - 1)
            nrows += 2 * grad_vol_size

        A = np.zeros((nrows, ncols), dtype=self.dtype)
        vol_mask = circular_mask([num_sino_pixels] * 2)

        for ii, bas_f in enumerate(self.basis_f):
            data_wu_f = self.apply_filter(data_wu, bas_f)

            img = self.projector.T(data_wu_f)
            img *= vol_mask

            A[:sino_size, ii] = self.projector(img).flatten()

            if self.lambda_smooth:
                dx = np.diff(img, axis=-2)
                dy = np.diff(img, axis=-1)
                d = np.concatenate((dx.flatten(), dy.flatten()))
                A[sino_size:, ii] = self.lambda_smooth * d

        b = np.zeros((nrows,), dtype=self.dtype)
        b[:sino_size] = data_wu.flatten()
        fitted_components = np.linalg.lstsq(A, b, rcond=None)[0].astype(self.dtype)

        self.fbp_filter = fitted_components.dot(self.basis_f)
