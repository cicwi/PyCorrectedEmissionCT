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
    FoV_size = 29
    add_poisson = True
    fit_reg_weight = False

    fitted_rec_weights_mc = np.array([152.261, 272.096])

    phantom = ph_gen.create_phantom_dots_overlap_2D(FoV_size)

    # Algorithms parameters
    iterations = 5000
    lower_limit = 0
    lambda_reg = 2e-1

    dwell_time_pencil = 1
    dwell_time_masked = 1
    beam_photons_pencil = 2e7
    beam_photons_masked = 2e9

    print("Pencil beam illumination:")
    mc_gen = cct.struct_illum.MaskGeneratorPoint([FoV_size, FoV_size])
    mc = mc_gen.generate_collection()
    p_pb = cct.struct_illum.ProjectorGhostImaging(mc)

    prj_info_pencil = data_gen.compute_buckets_multichannel(
        phantom,
        p_pb,
        voxel_size_um=0.5,
        dwell_time_s=dwell_time_pencil,
        add_poisson=add_poisson,
        beam_photons=beam_photons_pencil,
        do_attenuation=True,
    )

    num_channels = len(prj_info_pencil)

    print("\nReconstructions single-channel (%d):" % num_channels)
    phs_pb = [np.array([])] * num_channels
    recs_pb_ls = [np.array([])] * num_channels
    for ii_c in range(num_channels):
        bucket_vals_pencil = prj_info_pencil[ii_c].photon_counts

        phs_pb[ii_c] = prj_info_pencil[ii_c].expected_voxel_photon_counts

        recs_pb_ls[ii_c] = p_pb.bp(bucket_vals_pencil)

    # mc_gen = cct.struct_illum.MaskCollectionGeneratorMURA(FoV_size)
    mc_gen = cct.struct_illum.MaskGeneratorHalfGaussian([FoV_size, FoV_size])

    # fracs = [8, 4, 2, 1]
    fracs = [4, 2]
    recs_gi_xc = []
    recs_gi_ls = []
    recs_gi_tv = []

    reg_weights_sc = np.empty((len(fracs), num_channels))
    reg_weights_mc = np.empty((len(fracs)))

    for ii_f, frac in enumerate(fracs):
        print(f"\nFull FoV structured beam illumination with: {mc_gen.info().upper()} (1/{frac})")
        mc = mc_gen.generate_collection(buckets_fraction=1.0 / frac, shift_type="random")
        p_gi = cct.struct_illum.ProjectorGhostImaging(mc)

        prj_info_ghost = data_gen.compute_buckets_multichannel(
            phantom,
            p_gi,
            voxel_size_um=0.5,
            dwell_time_s=dwell_time_masked,
            add_poisson=add_poisson,
            beam_photons=beam_photons_masked,
            do_attenuation=True,
        )
        bucket_ones = np.ones_like(prj_info_ghost[0].photon_counts)
        bwd_prj_weights = p_gi.bp(bucket_ones) / bucket_ones.size

        print("\nReconstructions single-channel (%d):" % num_channels)
        imgs_ghost_ph = [np.array([])] * num_channels
        imgs_gi_xc = [np.array([])] * num_channels
        imgs_gi_ls = [np.array([])] * num_channels
        imgs_gi_tv = [np.array([])] * num_channels
        solvers_tv = [None] * num_channels

        for ii_c in range(num_channels):
            bucket_vals_ghost = prj_info_ghost[ii_c].photon_counts

            imgs_ghost_ph[ii_c] = prj_info_ghost[ii_c].expected_voxel_photon_counts

            theo_bucket_vals_masked = prj_info_ghost[ii_c].expected_photon_counts
            theo_phantom_vals_masked = prj_info_ghost[ii_c].expected_voxel_photon_counts

            dose_ratio = np.sum(prj_info_ghost[ii_c].incident_photons_pixel) / np.sum(
                prj_info_pencil[ii_c].incident_photons_pixel
            )
            print(f"Dose ratio: {dose_ratio} GI / PB")

            print("\nReconstructing '%s':" % prj_info_ghost[ii_c].signal_name)
            rec_gi_xc = p_gi.fbp(bucket_vals_ghost)
            imgs_gi_xc[ii_c] = np.squeeze(rec_gi_xc)

            sol_ls = cct.solvers.PDHG(verbose=True)
            rec_gi_ls, _ = sol_ls(p_gi, bucket_vals_ghost, iterations=25, lower_limit=lower_limit)

            imgs_gi_ls[ii_c] = np.squeeze(rec_gi_ls)

            # reg_type = lambda x, m: cct.regularizers.Regularizer_l12swl(x, "haar", 1, upd_mask=m)
            reg_type = lambda x, m: cct.regularizers.Regularizer_TV2D(x, upd_mask=m)

            if fit_reg_weight:
                reg_weight = rec_utils.find_reg_weight(
                    p_gi, bucket_vals_ghost, iterations, reg=reg_type, lambda_range=(1e2, 1e3)
                )
            else:
                reg_weight = lambda_reg * np.mean(bucket_vals_ghost) / np.sqrt(frac)

            reg_sc = reg_type(reg_weight, m=bwd_prj_weights)
            reg_weights_sc[ii_f, ii_c] = reg_weight

            pdhg_sc = cct.solvers.PDHG(verbose=True, regularizer=reg_sc)
            rec_gi_tv, _ = pdhg_sc(p_gi, bucket_vals_ghost, iterations=iterations, lower_limit=lower_limit)
            solvers_tv[ii_c] = pdhg_sc

            imgs_gi_tv[ii_c] = np.squeeze(rec_gi_tv)

        recs_gi_xc.append(imgs_gi_xc)
        recs_gi_ls.append(imgs_gi_ls)
        recs_gi_tv.append(imgs_gi_tv)

        print("\nReconstructions multi-channel:")
        bucket_vals_ghost = [x.photon_counts for x in prj_info_ghost]
        bucket_vals_ghost = np.stack(bucket_vals_ghost, axis=0)
        # Let's renormalize the channels
        norm_channels = bucket_vals_ghost.mean(axis=tuple(np.arange(1, bucket_vals_ghost.ndim)), keepdims=True)
        norm_channels /= norm_channels.mean()
        bucket_vals_ghost /= norm_channels

        # reg_type = lambda x, m: cct.regularizers.Regularizer_TNV(x, upd_mask=m)
        reg_type = lambda x, m: cct.regularizers.Regularizer_VTV(x, upd_mask=m)

        if fit_reg_weight:
            reg_weight = rec_utils.find_reg_weight(p_gi, bucket_vals_ghost, iterations, reg=reg_type, lambda_range=(1e2, 1e3))
        else:
            reg_weight = fitted_rec_weights_mc[ii_f]
            # reg_weight = lambda_reg * np.mean(bucket_vals_ghost) / np.sqrt(frac)

        reg_mc = reg_type(reg_weight, m=bwd_prj_weights)
        reg_weights_mc[ii_f] = reg_weight

        pdhg_mc = cct.solvers.PDHG(verbose=True, regularizer=reg_mc)
        imgs_gi_tv_mc, _ = pdhg_mc(p_gi, bucket_vals_ghost, iterations=iterations, lower_limit=lower_limit)

        imgs_gi_tv_mc *= norm_channels.reshape([-1, *np.ones_like(imgs_gi_tv_mc.shape[1:])])

        figsize = cm2inch(np.array([42, 7 * num_channels]) + [0, 1])
        f, axs = plt.subplots(num_channels, 6, sharex=True, sharey=True, figsize=figsize)
        dose_ratio = np.sum(prj_info_ghost[0].incident_photons_pixel) / np.sum(prj_info_pencil[0].incident_photons_pixel)
        time_ratio = np.sum(prj_info_ghost[0].total_measurement_time_s) / np.sum(prj_info_pencil[0].total_measurement_time_s)
        f.suptitle(f"Fraction: 1/{frac} - dose: x{dose_ratio}, time: x{time_ratio} - Iterations: {iterations}")
        for ii_c in range(num_channels):
            axs[ii_c, 0].imshow(np.squeeze(phs_pb[ii_c]))
            axs[ii_c, 0].set_title(f"Expected avg counts PB: {prj_info_pencil[ii_c].signal_name}")
            axs[ii_c, 1].imshow(np.squeeze(recs_pb_ls[ii_c]))
            axs[ii_c, 1].set_title("Acquired counts PB")
            axs[ii_c, 2].imshow(np.squeeze(imgs_ghost_ph[ii_c]))
            axs[ii_c, 2].set_title("Expected avg counts GI")
            axs[ii_c, 3].imshow(np.squeeze(imgs_gi_xc[ii_c]))
            axs[ii_c, 3].set_title("X-correlation GI")
            axs[ii_c, 4].imshow(np.squeeze(imgs_gi_tv[ii_c]))
            axs[ii_c, 4].set_title(solvers_tv[ii_c].info())
            axs[ii_c, 5].imshow(np.squeeze(imgs_gi_tv_mc[ii_c, ...]))
            axs[ii_c, 5].set_title(pdhg_mc.info())
        f.tight_layout()
        f.subplots_adjust(top=0.95)
        plt.show(block=False)
