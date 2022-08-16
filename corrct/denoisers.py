# -*- coding: utf-8 -*-
"""
Advanced denoising methods.

@author: Nicola VIGANÒ, Computational Imaging group, CWI, The Netherlands,
and ESRF - The European Synchrotron, Grenoble, France
"""

import numpy as np
import scipy.signal as spsig

from . import operators
from . import data_terms
from . import regularizers
from . import solvers
from . import utils_reg
from . import filters
from . import utils_proc

import matplotlib.pyplot as plt

try:
    from . import utils_nn

    has_nn = True

except ImportError as ex:
    print(ex)

    has_nn = False

from typing import List, Optional, Sequence, Tuple, Union, Callable
from numpy.typing import ArrayLike, NDArray


eps = np.finfo(np.float32).eps


def denoise_image(
    img: NDArray,
    reg_weight: Union[float, ArrayLike, NDArray] = 1e-2,
    psf: Union[ArrayLike, NDArray, None] = None,
    pix_weights: Optional[NDArray] = None,
    iterations: int = 250,
    regularizer: Callable = lambda rw: regularizers.Regularizer_l1dwl(rw, "bior4.4", 3),
    lower_limit: Optional[float] = None,
    verbose: bool = False,
) -> NDArray:
    """
    Denoise an image.

    Image denoiser based on (flat or weighted) least-squares, with wavelet minimization regularization.
    The weighted least-squares requires the local pixel-wise weights.
    It can be used to denoise sinograms and projections.

    Parameters
    ----------
    img : NDArray
        The image or sinogram to denoise.
    reg_weight : Union[float, ArrayLike, NDArray], optional
        Weight of the regularization term. The default is 1e-2.
        If a sequence / array is passed, all the different values will be tested.
        The one minimizing the error over the cross-validation set will be chosen and returned.
    pix_weights : Union[ArrayLike, NDArray, None], optional
        The local weights of the pixels, for a weighted least-squares minimization.
        If None, a standard least-squares minimization is performed. The default is None.
    iterations : int, optional
        Number of iterations. The default is 250.
    regularizer : Callable, optional
        The one-argument constructor of a regularizer. The default is the DWL regularizer.
    lower_limit : Optional[float], optional
        Lower clipping limit of the image. The default is None.
    verbose : bool, optional
        Turn verbosity on. The default is False.

    Returns
    -------
    NDArray
        Denoised image or sinogram.
    """
    if psf is None:
        op = operators.TransformIdentity(img.shape)
    else:
        op = operators.TransformConvolution(img.shape, psf)

    if pix_weights is None:
        data_term = data_terms.DataFidelity_l2()
    else:
        data_term = data_terms.DataFidelity_wl2(pix_weights)

    def solver_spawn(lam_reg):
        # Using the PDHG solver from Chambolle and Pock
        reg = regularizer(lam_reg)
        return solvers.PDHG(verbose=verbose, data_term=data_term, regularizer=reg, data_term_test=data_term)

    def solver_call(solver: solvers.Solver, b_test_mask: Optional[NDArray] = None) -> Tuple[NDArray, solvers.SolutionInfo]:
        x0 = img.copy()
        if b_test_mask is not None:
            med_img = spsig.medfilt2d(img, kernel_size=11)
            masked_pixels = b_test_mask > 0.5

            x0[masked_pixels] = med_img[masked_pixels]

        return solver(op, img, iterations, x0=x0, lower_limit=lower_limit, b_test_mask=b_test_mask)

    reg_weight = np.array(reg_weight)
    if reg_weight.size > 1:
        reg_help_cv = utils_reg.CrossValidation(img.shape, verbose=True, num_averages=5, plot_result=False)
        reg_help_cv.solver_spawning_function = solver_spawn
        reg_help_cv.solver_calling_function = solver_call

        f_avgs, f_stds, _ = reg_help_cv.compute_loss_values(reg_weight)

        reg_help_cv.plot_result = True
        reg_weight, _ = reg_help_cv.fit_loss_min(reg_weight, f_avgs, f_stds=f_stds)

    solver = solver_spawn(reg_weight)
    (denoised_img, _) = solver_call(solver, None)

    return denoised_img


if has_nn:

    class Denoiser_NNFBP:
        """Denoise FBP using Neural Networks."""

        projector: operators.ProjectorOperator
        num_fbps: int
        hidden_layers: List[int]

        def __init__(
            self,
            projector: operators.ProjectorOperator,
            num_fbps: int = 4,
            num_pixels_trn: int = 256,
            num_pixels_tst: int = 128,
            hidden_layers: List[int] = list(),
            verbose: bool = False,
        ) -> None:
            self.projector = projector
            self.num_fbps = num_fbps
            self.num_pixels_trn = num_pixels_trn
            self.num_pixels_tst = num_pixels_tst
            self.hidden_layers = hidden_layers
            self.verbose = verbose

            self.filters = filters.FilterCustom([1.0])

        def _get_normalization(self, vol: NDArray, percentile: Optional[float] = None) -> Tuple[float, float]:
            if percentile is not None:
                vol_sort = np.sort(vol.flatten())
                ind_min = np.fmax(int(vol_sort.size * percentile), 0)
                ind_max = np.fmin(int(vol_sort.size * (1 - percentile)), vol_sort.size - 1)
                return vol_sort[ind_min], vol_sort[ind_max]
            else:
                return vol.min(), vol.max()

        def _get_pixel_subsampling(
            self, data_shape: Sequence[int], *, num_trn: int, num_tst: int, num_vld: int = 0
        ) -> Sequence[NDArray]:
            vol_mask = utils_proc.get_circular_mask(data_shape)
            vol_linear_pos = np.arange(vol_mask.size)[vol_mask.flatten() == 1.0]

            num_pixels = num_trn + num_tst + num_vld

            pixels_pos = np.random.permutation(int(np.sum(vol_mask)))[:num_pixels]
            pixels_pos_trn = pixels_pos[:num_trn]
            pixels_pos_tst = pixels_pos[num_trn : num_trn + num_tst]

            pixels_pos_trn = vol_linear_pos[pixels_pos_trn]
            pixels_pos_tst = vol_linear_pos[pixels_pos_tst]

            if num_vld == 0:
                return pixels_pos_trn, pixels_pos_tst
            else:
                pixels_pos_vld = pixels_pos[:-num_vld]
                pixels_pos_vld = vol_linear_pos[pixels_pos_vld]

                return pixels_pos_trn, pixels_pos_tst, pixels_pos_vld

        def _sub_sample_pixels(
            self, pixels_pos: NDArray, inp_recs: List[NDArray], tgt_recs: NDArray, tgt_wgts: Optional[NDArray] = None
        ) -> Sequence[NDArray]:
            X = np.stack([inp_rec.flatten()[pixels_pos] for inp_rec in inp_recs], axis=-1)
            y = tgt_recs.flatten()[pixels_pos]

            if tgt_wgts is None:
                return X, y
            else:
                w = tgt_wgts.flatten()[pixels_pos]
                return X, y, w

        def fit(
            self,
            inp_sinos: NDArray,
            tgt_recs: NDArray,
            tgt_wgts: Optional[NDArray] = None,
            train_epochs: int = 10_000,
            init_fit_weights: Optional[Tuple[Sequence[NDArray], Sequence[NDArray]]] = None,
            plot_loss_curves: bool = True,
        ) -> None:
            num_pixels = self.filters.get_padding_size(inp_sinos.shape)
            self.basis_r = filters.create_basis(num_pixels, binning_start=3, binning_type="exponential", normalized=True)
            self.basis_f = self.filters.to_fourier(self.basis_r)
            self.filters = filters.FilterCustom(self.basis_f)

            num_filters = self.filters.num_filters

            inp_sinos_filt = self.filters.apply_filter(inp_sinos)
            inp_recs = [np.array([])] * num_filters
            for ii in range(num_filters):
                inp_recs[ii] = self.projector.T(inp_sinos_filt[ii])

            # Compute scaling
            min_max_features = [self._get_normalization(lq_rec, percentile=0.001) for lq_rec in inp_recs]
            self.min_max_target = self._get_normalization(tgt_recs)

            self.basis_f_scaled = [
                filt / (max - min) for filt, (min, max) in zip(self.filters.filter_fourier, min_max_features)
            ]
            inp_recs = [inp_rec / (max - min) for inp_rec, (min, max) in zip(inp_recs, min_max_features)]
            tgt_recs = tgt_recs / (self.min_max_target[1] - self.min_max_target[0])

            pix_trn, pix_tst = self._get_pixel_subsampling(
                tgt_recs.shape, num_trn=self.num_pixels_trn, num_tst=self.num_pixels_tst
            )
            data_trn = self._sub_sample_pixels(pix_trn, inp_recs, tgt_recs, tgt_wgts=tgt_wgts)
            data_tst = self._sub_sample_pixels(pix_tst, inp_recs, tgt_recs, tgt_wgts=tgt_wgts)

            self.nn_fit = utils_nn.NeuralNetwork(layers_size=[num_filters, self.num_fbps, *self.hidden_layers, 1])
            if init_fit_weights is not None:
                self.nn_fit.model.set_weights(weights=init_fit_weights[0], biases=init_fit_weights[1])

            info_a = self.nn_fit.train_adam(data_trn, data_tst, iterations=2 * train_epochs)
            info_n = self.nn_fit.train_lbfgs(data_trn, data_tst, iterations=train_epochs)

            if plot_loss_curves:
                f, axs = plt.subplots(1, 2, figsize=[10, 4])
                axs[0].semilogy(info_a.loss_values_trn, label="Train loss")
                axs[0].semilogy(np.array([info_a.loss_init_tst, *info_a.loss_values_tst]), label="Test loss")
                axs[1].semilogy(info_n.loss_values_trn, label="Train loss")
                axs[1].semilogy(np.array([info_n.loss_init_tst, *info_n.loss_values_tst]), label="Test loss")
                axs[0].grid()
                axs[1].grid()
                axs[1].legend()
                f.tight_layout()
                plt.show(block=False)

            self._create_prediction_nn()

        def _create_prediction_nn(self):
            w, b = self.nn_fit.model.get_weights()
            filters_learned = w[0].dot(self.basis_f_scaled)

            self.filters = filters.FilterCustom(filters_learned)
            w[0] = np.eye(self.num_fbps)

            self.nn_predict = utils_nn.NeuralNetwork(layers_size=[self.num_fbps, self.num_fbps, *self.hidden_layers, 1])
            self.nn_predict.model.set_weights(w, b)

        def predict(self, inp_sinos: NDArray) -> NDArray:
            filt_sinos = self.filters.apply_filter(inp_sinos)
            filt_recs = [np.array([])] * self.filters.num_filters
            for ii in range(self.filters.num_filters):
                filt_recs[ii] = self.projector.T(filt_sinos[ii])

            filt_recs_stack = np.stack([filt_rec.flatten() for filt_rec in filt_recs], axis=-1)

            pred_rec = self.nn_predict.predict([filt_recs_stack])
            pred_rec *= self.min_max_target[1] - self.min_max_target[0]
            return pred_rec.reshape(filt_recs[0].shape)

    class Denoiser_N2F(Denoiser_NNFBP):
        """Denoise FBP using Neural Networks, with a Noise2Inverse strategy."""

        def _get_pixel_subsampling(
            self, data_shape: Sequence[int], *, num_trn: int, num_tst: int, num_vld: int = 0
        ) -> Sequence[NDArray]:
            def expand_multiple_vols(inds: NDArray, num_vols: int, size_vol: int) -> NDArray:
                return np.concatenate([inds + ii * size_vol for ii in range(num_vols)])

            num_tgts = data_shape[0]
            vol_mask = utils_proc.get_circular_mask(data_shape[1:])
            vol_linear_pos = np.arange(vol_mask.size)[vol_mask.flatten() == 1.0]

            num_pixels = num_trn + num_tst + num_vld

            pixels_pos = np.random.permutation(int(np.sum(vol_mask)))[:num_pixels]
            pixels_pos_trn = pixels_pos[:num_trn]
            pixels_pos_tst = pixels_pos[num_trn : num_trn + num_tst]

            pixels_pos_trn = expand_multiple_vols(vol_linear_pos[pixels_pos_trn], num_tgts, vol_mask.size)
            pixels_pos_tst = expand_multiple_vols(vol_linear_pos[pixels_pos_tst], num_tgts, vol_mask.size)

            if num_vld == 0:
                return pixels_pos_trn, pixels_pos_tst
            else:
                pixels_pos_vld = pixels_pos[:-num_vld]
                pixels_pos_vld = expand_multiple_vols(vol_linear_pos[pixels_pos_vld], num_tgts, vol_mask.size)

                return pixels_pos_trn, pixels_pos_tst, pixels_pos_vld

        def fit(
            self,
            inp_sinos: NDArray,
            tgt_recs: NDArray,
            tgt_wgts: Optional[NDArray] = None,
            train_epochs: int = 10_000,
            init_fit_weights: Optional[Tuple[Sequence[NDArray], Sequence[NDArray]]] = None,
            projectors: Optional[Sequence[operators.ProjectorOperator]] = None,
            plot_loss_curves: bool = True,
        ) -> None:
            num_sinos = inp_sinos.shape[0]
            if projectors is not None and num_sinos != len(projectors):
                raise ValueError(
                    f"Number of sinograms ({num_sinos}) and number of projectors ({len(projectors)}) should be equal."
                )
            num_angles = inp_sinos.shape[-2]
            num_pixels = self.filters.get_padding_size(inp_sinos.shape[1:])

            self.basis_r = filters.create_basis(num_pixels, binning_start=3, binning_type="exponential", normalized=True)
            self.basis_f = self.filters.to_fourier(self.basis_r)
            self.filters = filters.FilterCustom(self.basis_f)

            num_filters = self.filters.num_filters

            inp_sinos_filt = self.filters.apply_filter(inp_sinos)
            inp_recs = [np.array([])] * num_filters
            for ii in range(num_filters):
                if projectors is None:
                    inp_recs[ii] = np.ascontiguousarray([self.projector.T(s) for s in inp_sinos_filt[ii]])
                else:
                    inp_recs[ii] = np.ascontiguousarray([projectors[ii_s].T(s) for ii_s, s in enumerate(inp_sinos_filt[ii])])

            # Compute scaling
            min_max_features = [self._get_normalization(np.mean(inp_rec, axis=0), percentile=0.001) for inp_rec in inp_recs]
            self.min_max_target = self._get_normalization(np.mean(tgt_recs, axis=0), percentile=0.001)

            self.basis_f_scaled = np.ascontiguousarray(
                [filt / (max - min) for filt, (min, max) in zip(self.filters.filter_fourier, min_max_features)]
            )
            if projectors is not None:
                # We need to normalize the learned filters by the number of angles of the projector to be used.
                # We assume that the projectors are partitions of the main projector.
                self.basis_f_scaled /= self.projector.prj_shape[-2] / num_angles

            inp_recs = [inp_rec / (max - min) for inp_rec, (min, max) in zip(inp_recs, min_max_features)]
            tgt_recs = tgt_recs / (self.min_max_target[1] - self.min_max_target[0])

            pix_trn, pix_tst = self._get_pixel_subsampling(
                tgt_recs.shape, num_trn=self.num_pixels_trn, num_tst=self.num_pixels_tst
            )
            data_trn = self._sub_sample_pixels(pix_trn, inp_recs, tgt_recs, tgt_wgts=tgt_wgts)
            data_tst = self._sub_sample_pixels(pix_tst, inp_recs, tgt_recs, tgt_wgts=tgt_wgts)

            self.nn_fit = utils_nn.NeuralNetwork(layers_size=[num_filters, self.num_fbps, *self.hidden_layers, 1])
            if init_fit_weights is not None:
                self.nn_fit.model.set_weights(weights=init_fit_weights[0], biases=init_fit_weights[1])

            info_a = self.nn_fit.train_adam(data_trn, data_tst, iterations=5 * train_epochs)
            info_n = self.nn_fit.train_lbfgs(data_trn, data_tst, iterations=train_epochs)

            if plot_loss_curves:
                f, axs = plt.subplots(1, 2, figsize=[10, 4])
                axs[0].semilogy(info_a.loss_values_trn, label="Train loss")
                axs[0].semilogy(np.array([info_a.loss_init_tst, *info_a.loss_values_tst]), label="Test loss")
                axs[1].semilogy(info_n.loss_values_trn, label="Train loss")
                axs[1].semilogy(np.array([info_n.loss_init_tst, *info_n.loss_values_tst]), label="Test loss")
                axs[0].grid()
                axs[1].grid()
                axs[1].legend()
                f.tight_layout()
                plt.show(block=False)

            self._create_prediction_nn()
