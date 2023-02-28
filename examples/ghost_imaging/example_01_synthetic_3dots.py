# -*- coding: utf-8 -*-
"""
Ghost-imaging reconstruction example.

@author: Nicola VIGANÒ, The European Synchrotron, Grenoble, France
"""

import numpy as np

import matplotlib.pyplot as plt

import corrct as cct

# local packages
import phantom_creation as ph_gen
import data_generation as data_gen
import reconstruction_utils as rec_utils

from typing import Sequence, Union
from numpy.typing import NDArray


def cm2inch(x: Union[float, Sequence[float], NDArray]) -> Sequence[float]:
    """Convert centimeters to inches.

    Parameters
    ----------
    x : float
        length in cm.

    Returns
    -------
    float
        length in inch.
    """
    return list(np.array(x) / 2.54)


if __name__ == "__main__":
    FoV_size = 37
    add_poisson = True
    fit_reg_weight = False

    phantom = ph_gen.create_phantom_dots_2D(FoV_size)

    # Algorithms parameters
    iterations = 5000
    lower_limit = 0
    lambda_reg = 1e-2
    # lambda_reg = 1e4  # To be used in the noiseless case

    dwell_time_pencil = 1
    dwell_time_masked = 1
    beam_photons_pencil = 1e9
    beam_photons_masked = 1e11

    print("Pencil beam illumination:")
    mc_gen = cct.struct_illum.MaskGeneratorPoint([FoV_size, FoV_size])
    mc = mc_gen.generate_collection()
    p_pb = cct.struct_illum.ProjectorGhostImaging(mc)

    prj_info_pencil = data_gen.compute_buckets(
        phantom,
        p_pb,
        voxel_size_um=0.5,
        dwell_time_s=dwell_time_pencil,
        add_poisson=add_poisson,
        beam_photons=beam_photons_pencil,
    )

    imgs_pencil_ph = prj_info_pencil.expected_voxel_photon_counts
    imgs_pencil_ls = p_pb.bp(prj_info_pencil.photon_counts)

    mc_gens: Sequence[cct.struct_illum.MaskGenerator] = [
        # cct.struct_illum.MaskCollectionGeneratorMURA(FoV_size),
        # cct.struct_illum.MaskGeneratorBernoulli([FoV_size, FoV_size]),
        cct.struct_illum.MaskGeneratorHalfGaussian([FoV_size, FoV_size]),
    ]
    # fracs = [8, 4, 2, 1]
    fracs = [16, 8, 4, 2]
    recs_xc = []
    recs_ls = []
    recs_reg = []

    phs = []

    reg_weights = np.empty((len(mc_gens), len(fracs)))

    for ii_m, mc_gen in enumerate(mc_gens):
        for ii_f, frac in enumerate(fracs):
            print(f"\nFull FoV structured beam illumination with: {mc_gen.info().upper()} (1/{frac})")
            mc = mc_gen.generate_collection(buckets_fraction=1.0 / frac, shift_type="random")

            p_gi = cct.struct_illum.ProjectorGhostImaging(mc)

            prj_info_ghost = data_gen.compute_buckets(
                phantom,
                p_gi,
                voxel_size_um=0.5,
                dwell_time_s=dwell_time_masked,
                add_poisson=add_poisson,
                beam_photons=beam_photons_masked,
            )
            bucket_ones = np.ones_like(prj_info_ghost.photon_counts)
            bwd_prj_weights = p_gi.bp(bucket_ones)
            bwd_prj_weights /= bwd_prj_weights.mean()

            bucket_vals_gi = prj_info_ghost.photon_counts

            # Expected reconstructions
            imgs_ghost_ph = prj_info_ghost.expected_voxel_photon_counts

            # Dose ratio between pencil beam, and ghost imaging
            dose_ratio = np.sum(prj_info_ghost.incident_photons_pixel) / np.sum(prj_info_pencil.incident_photons_pixel)
            print(f"Dose ratio: {dose_ratio} GI / PB")

            rec_img_xc = p_gi.fbp(bucket_vals_gi)

            sol_ls = cct.solvers.PDHG(verbose=True)
            rec_img_ls, _ = sol_ls(p_gi, bucket_vals_gi, iterations=25, lower_limit=lower_limit)

            reg_type = lambda x, m: cct.regularizers.Regularizer_l12swl(x, "haar", 1, upd_mask=m)
            # reg_type = lambda x, m: cct.regularizers.Regularizer_TV2D(x, upd_mask=m)

            if fit_reg_weight:
                reg_weight = rec_utils.find_reg_weight(p_gi, bucket_vals_gi, iterations, reg=reg_type, lambda_range=(1e2, 1e4))
            else:
                reg_weight = lambda_reg * bucket_vals_gi.mean() / np.sqrt(frac)

            reg = reg_type(reg_weight, m=bwd_prj_weights)
            reg_weights[ii_m, ii_f] = reg_weight

            sol_tv = cct.solvers.PDHG(verbose=True, regularizer=reg)
            rec_img_reg, _ = sol_tv(p_gi, bucket_vals_gi, iterations=iterations, lower_limit=lower_limit)

            recs_xc.append(np.squeeze(rec_img_xc))
            recs_ls.append(np.squeeze(rec_img_ls))
            recs_reg.append(np.squeeze(rec_img_reg))
            phs.append(np.squeeze(imgs_ghost_ph))

    recs_xc = np.stack(recs_xc)
    recs_xc = recs_xc.reshape([len(mc_gens), len(fracs), *recs_xc.shape[-2:]])

    recs_ls = np.stack(recs_ls)
    recs_ls = recs_ls.reshape([len(mc_gens), len(fracs), *recs_ls.shape[-2:]])

    recs_reg = np.stack(recs_reg)
    recs_reg = recs_reg.reshape([len(mc_gens), len(fracs), *recs_reg.shape[-2:]])

    phs = np.stack(phs)
    phs = phs.reshape([len(mc_gens), len(fracs), *phs.shape[-2:]])

    max_val = np.max([np.sort(recs_reg.flatten())[-10], np.sort(phs.flatten())[-10]])

    figsize = cm2inch(np.array([len(fracs) + 1, len(mc_gens)]) * 10 + [0, 2])
    f, axs = plt.subplots(len(mc_gens), len(fracs) + 1, sharex=True, sharey=True, squeeze=False, figsize=figsize)
    f.suptitle("Comparison of masks")
    for ii_m, mc_gen in enumerate(mc_gens):
        mc = mc_gen.generate_collection()
        for ii_f, frac in enumerate(fracs):
            axs[ii_m, ii_f].imshow(recs_reg[ii_m, ii_f, ...], vmax=max_val)  #
            axs[ii_m, ii_f].set_title(f"{mc.upper()} - 1/{frac} points")
        axs[ii_m, -1].imshow(phs[ii_m, 0], vmax=max_val)  #
        axs[ii_m, -1].set_title(f"Expected (Noiseless)")
    f.tight_layout()
    f.subplots_adjust(top=0.95)
    plt.show(block=False)
