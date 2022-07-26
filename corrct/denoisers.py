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

        projector: operators.BaseTransform
        num_fbps: int
        hidden_layers: List[int]

        def __init__(
            self,
            projector: operators.BaseTransform,
            num_fbps: int = 4,
            num_pixels_train: int = 256,
            num_pixels_test: int = 128,
            hidden_layers: List[int] = list(),
        ) -> None:
            self.projector = projector
            self.num_fbps = num_fbps
            self.num_pixels_train = num_pixels_train
            self.num_pixels_test = num_pixels_test
            self.hidden_layers = hidden_layers

            self.filters = filters.FilterCustom([1.0])

        def _get_normalization(self, vol: NDArray, percentile: Optional[float] = None) -> Tuple[float, float]:
            if percentile is not None:
                vol_sort = np.sort(vol.flatten())
                ind_min = np.fmax(int(vol_sort.size * percentile), 0)
                ind_max = np.fmin(int(vol_sort.size * (1 - percentile)), vol_sort.size - 1)
                return vol_sort[ind_min], vol_sort[ind_max]
            else:
                return vol.min(), vol.max()

        def _sub_sample_pixels(
            self, num_pixels: int, lq_recs: List[NDArray], hq_recs: NDArray, hq_weights: Optional[NDArray] = None
        ) -> utils_nn.DatasetPixel:
            vol_mask = utils_proc.get_circular_mask(hq_recs.shape)
            vol_linear_pos = np.arange(vol_mask.size)[vol_mask.flatten() == 1.0]

            pixels_pos = np.random.permutation(int(np.sum(vol_mask)))[:num_pixels]
            pixels_pos = vol_linear_pos[pixels_pos]

            X = np.stack([lq_rec.flatten()[pixels_pos] for lq_rec in lq_recs], axis=-1)
            y = hq_recs.flatten()[pixels_pos]

            if hq_weights is None:
                return utils_nn.DatasetPixel(X, y)
            else:
                w = hq_weights.flatten()[pixels_pos]
                return utils_nn.DatasetPixel(X, y, w)

        def fit(
            self,
            inp_sinos: NDArray,
            tgt_recs: NDArray,
            train_epochs: int = 10_000,
            init_fit_weights: Optional[Tuple[Sequence[NDArray], Sequence[NDArray]]] = None,
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

            inp_recs = [inp_rec / (max - min) for inp_rec, (min, max) in zip(inp_recs, min_max_features)]
            tgt_recs = tgt_recs / (self.min_max_target[1] - self.min_max_target[0])

            dset_train = self._sub_sample_pixels(self.num_pixels_train, inp_recs, tgt_recs)
            dset_test = self._sub_sample_pixels(self.num_pixels_test, inp_recs, tgt_recs)

            self.nn_fit = utils_nn.NeuralNetwork(layers_size=[num_filters, self.num_fbps, *self.hidden_layers, 1])
            if init_fit_weights is not None:
                self.nn_fit.model.set_weights(weights=init_fit_weights[0], biases=init_fit_weights[1])
            self.nn_fit.train(dset_train, dataset_test=dset_test, iterations=train_epochs)

            w, b = self.nn_fit.model.get_weights()
            filters_scaled = [filt / (max - min) for filt, (min, max) in zip(self.filters.filter_fourier, min_max_features)]
            filters_learned = w[0].dot(filters_scaled)

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
            dset_predict = utils_nn.DatasetPixel(filt_recs_stack)

            pred_rec = self.nn_predict.predict(dset_predict)
            pred_rec *= self.min_max_target[1] - self.min_max_target[0]
            return pred_rec.reshape(filt_recs[0].shape)
