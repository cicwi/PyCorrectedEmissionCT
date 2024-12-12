# -*- coding: utf-8 -*-
"""
Ghost-imaging phantom creation.

@author: Nicola VIGANÒ, The European Synchrotron, Grenoble, France
"""

import numpy as np

import corrct as cct

from dataclasses import dataclass

from numpy.typing import NDArray
from typing import Sequence, Union

from tqdm.auto import tqdm


@dataclass
class ProjectionData:
    """Projected values info."""

    signal_name: str
    photon_counts: NDArray
    num_incident_photons_beam: float
    total_measurement_time_s: float
    photon_density_per_pixel: float
    incident_photons_pixel: NDArray
    expected_voxel_photon_counts: NDArray
    expected_photon_counts: NDArray


def compute_buckets(
    phantom: NDArray,
    projector: cct.struct_illum.ProjectorGhostImaging,
    element: str = "K",
    fluo_lines: str = "KA",
    energy_in_keV: float = 17,
    voxel_size_um: float = 0.5,
    dwell_time_s: float = 1,
    beam_photons: float = 1e9,
    background_avg: float = 1e-5,
    add_poisson: bool = False,
) -> ProjectionData:
    print("- Dwell time: %e seconds, " % dwell_time_s)
    beam_waist_vox = projector.mc.mask_support
    beam_waist_um = voxel_size_um * np.array(beam_waist_vox)
    print(
        "- Beam waist: [%d, %d] pix, [%f, %f] um" % (beam_waist_vox[0], beam_waist_vox[1], beam_waist_um[0], beam_waist_um[1])
    )
    total_measurement_time_s = dwell_time_s * projector.mc.masks_enc.shape[0]
    hours, remainder = divmod(total_measurement_time_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("- Total exposure time: {:02}h{:02}m{:02}s".format(int(hours), int(minutes), int(seconds)))
    conv_mm_to_cm = 1e-1
    conv_um_to_mm = 1e-3
    voxel_size_cm = voxel_size_um * conv_um_to_mm * conv_mm_to_cm  # cm to micron

    detector = cct.physics.xrf.DetectorXRF(distance_mm=12.0, surface_mm2=6.0)

    phantom_obj = cct.physics.VolumeMaterial((phantom,), ("Potassium Oxide",), voxel_size_cm)
    _, phantom_yield = phantom_obj.get_fluo_yield(
        element=element, energy_in_keV=energy_in_keV, fluo_lines=fluo_lines, detector=detector
    )

    num_incident_photons_beam = beam_photons * dwell_time_s
    photon_density_per_pixel = float(num_incident_photons_beam / np.prod(beam_waist_vox))
    print("- Average incident photons per beam: %e" % num_incident_photons_beam)

    incident_photons_pixel = photon_density_per_pixel * projector.col_sum
    print("- Average incident photons per pixel: %e" % np.mean(incident_photons_pixel))

    phantom_photons_expected = photon_density_per_pixel * phantom_yield
    buckets_yield = projector.fp(phantom_photons_expected)

    buckets_photons_measured, buckets_photons_expected, _ = cct.testing.add_noise(
        buckets_yield, 1, add_poisson=add_poisson, background_avg=background_avg
    )
    print("- Average photon counts per reading: %e" % np.mean(buckets_photons_expected))

    prj_info = ProjectionData(
        signal_name="%s K-alpha" % element,
        photon_counts=buckets_photons_measured,
        num_incident_photons_beam=num_incident_photons_beam,
        total_measurement_time_s=total_measurement_time_s,
        photon_density_per_pixel=photon_density_per_pixel,
        incident_photons_pixel=incident_photons_pixel,
        expected_voxel_photon_counts=phantom_photons_expected,
        expected_photon_counts=buckets_photons_expected,
    )
    return prj_info


def compute_buckets_3D(
    phantom_3d: NDArray,
    projector_gi: cct.struct_illum.ProjectorGhostImaging,
    angles_rot_deg: Union[Sequence[float], NDArray] = np.linspace(0, 180, 60),
    element: str = "Zn",
    fluo_lines: str = "KA",
    energy_in_keV: float = 17,
    voxel_size_um: Union[NDArray, Sequence[float]] = (0.5, 0.5, 0.5),
    dwell_time_s: float = 1,
    beam_photons: float = 1e9,
    background_avg: float = 1e-5,
    add_poisson: bool = False,
) -> ProjectionData:
    conv_mm_to_cm = 1e-1
    conv_um_to_mm = 1e-3
    voxel_size_cm = np.array(voxel_size_um) * conv_um_to_mm * conv_mm_to_cm  # cm to micron

    phantom_P = phantom_3d[0]  # Selecting membranes
    phantom_Zn = phantom_3d[1]  # Selecting nuclei
    phantom_C02 = phantom_3d[2]  # Water content

    detector = cct.physics.xrf.DetectorXRF(distance_mm=12.0, surface_mm2=6.0)

    # Tomo data creation
    cmp_p = cct.physics.get_compound("P")
    cmp_zn = cct.physics.get_compound("Zn")
    phantom_obj = cct.physics.VolumeMaterial(
        [phantom_P, phantom_Zn, phantom_C02], [cmp_p, cmp_zn, "Water, Liquid"], voxel_size_cm[-1]
    )
    energy_out_keV, phantom_Zn_fluo_yield = phantom_obj.get_fluo_yield(
        element, energy_in_keV=energy_in_keV, fluo_lines=fluo_lines, detector=detector
    )
    local_att_in = phantom_obj.get_attenuation(energy_in_keV)
    local_att_out = phantom_obj.get_attenuation(energy_out_keV)

    angles_rot_rad = np.deg2rad(angles_rot_deg)
    att = cct.physics.attenuation.AttenuationVolume(local_att_in, local_att_out, angles_rot_rad=angles_rot_rad)
    att.compute_maps()

    vol_geom = cct.models.VolumeGeometry.get_default_from_volume(phantom_Zn)
    with cct.projectors.ProjectorAttenuationXRF(vol_geom, angles_rot_rad, **att.get_projector_args()) as proj_tomo:
        prj_data_Zn = proj_tomo(phantom_Zn_fluo_yield)

    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[7, 3.75])
    # extent = att.plot_map(axs[1], rot_ind=0, slice_ind=12)
    # axs[0].imshow(phantom_Zn_fluo_yield[12], extent=extent)
    # fig.tight_layout()
    # plt.show(block=True)

    print("- Dwell time: %e seconds, " % dwell_time_s)
    beam_waist_vox = projector_gi.mc.mask_support
    beam_waist_um = voxel_size_um[-1] * np.array(beam_waist_vox)
    print(
        "- Beam waist: [%d, %d] pix, [%f, %f] um" % (beam_waist_vox[0], beam_waist_vox[1], beam_waist_um[0], beam_waist_um[1])
    )
    total_measurement_time_s = dwell_time_s * projector_gi.mc.masks_enc.shape[0]
    hours, remainder = divmod(total_measurement_time_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("- Total exposure time per image: {:02}h{:02}m{:02}s".format(int(hours), int(minutes), int(seconds)))

    num_incident_photons_beam = beam_photons * dwell_time_s
    photon_density_per_pixel = float(num_incident_photons_beam / np.prod(beam_waist_vox))
    print("- Average incident photons per beam: %e" % num_incident_photons_beam)

    incident_photons_pixel = photon_density_per_pixel * projector_gi.col_sum
    print("- Average incident photons per pixel: %e" % np.mean(incident_photons_pixel))

    phantom_photons_expected = photon_density_per_pixel * prj_data_Zn

    buckets_yield = [
        projector_gi.fp(phantom_photons_expected[..., ii, :])
        for ii in tqdm(range(len(angles_rot_rad)), desc="- Computing GI projections")
    ]
    buckets_yield = np.stack(buckets_yield)

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(np.squeeze(prj_data_Zn[..., 0, :]))
    # ax[1].plot(np.squeeze(buckets_yield[0]))
    # fig.tight_layout()
    # plt.show(block=True)

    buckets_photons_measured, buckets_photons_expected, _ = cct.testing.add_noise(
        buckets_yield, 1, add_poisson=add_poisson, background_avg=background_avg
    )
    print("- Average photon counts per reading: %e" % np.mean(buckets_photons_expected))

    prj_info = ProjectionData(
        signal_name="%s K-alpha" % element,
        photon_counts=buckets_photons_measured,
        num_incident_photons_beam=num_incident_photons_beam,
        total_measurement_time_s=total_measurement_time_s,
        photon_density_per_pixel=photon_density_per_pixel,
        incident_photons_pixel=incident_photons_pixel,
        expected_voxel_photon_counts=phantom_photons_expected,
        expected_photon_counts=buckets_photons_expected,
    )
    return prj_info


def compute_buckets_multichannel(
    phantom: NDArray,
    projector: cct.struct_illum.ProjectorGhostImaging,
    elements: Sequence[str] = ["Ca", "Fe", "Cu"],
    voxel_size_um: float = 0.5,
    dwell_time_s: float = 1,
    beam_photons: float = 1e9,
    background_avg: float = 1e-5,
    add_poisson: bool = False,
    do_compton: bool = False,
    do_attenuation: bool = False,
) -> Sequence[ProjectionData]:
    print("- Dwell time: %e seconds, " % dwell_time_s)
    beam_waist_vox = projector.mc.mask_support
    beam_waist_um = voxel_size_um * np.array(beam_waist_vox)
    print(
        "- Beam waist: [%d, %d] pix, [%f, %f] um" % (beam_waist_vox[0], beam_waist_vox[1], beam_waist_um[0], beam_waist_um[1])
    )
    total_measurement_time_s = dwell_time_s * projector.mc.masks_enc.shape[0]
    hours, remainder = divmod(total_measurement_time_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("- Total exposure time: {:02}h{:02}m{:02}s".format(int(hours), int(minutes), int(seconds)))
    conv_mm_to_cm = 1e-1
    conv_um_to_mm = 1e-3
    voxel_size_cm = voxel_size_um * conv_um_to_mm * conv_mm_to_cm  # cm to micron

    num_incident_photons_beam = beam_photons * dwell_time_s
    photon_density_per_pixel = float(num_incident_photons_beam / np.prod(beam_waist_vox))
    print("- Average incident photons per beam: %e" % num_incident_photons_beam)

    detector = cct.physics.xrf.DetectorXRF(distance_mm=34.0, surface_mm2=6.0)

    incident_photons_pixel = photon_density_per_pixel * projector.col_sum
    print("- Average incident photons per pixel: %e" % np.mean(incident_photons_pixel))

    cu2o = cct.physics.xraylib_helper.get_compound("Cu2O")
    cu2o["name"] = "Cuprous Oxide"
    cu2o["density"] = 6.0
    phase_compounds = ("Air, Dry (near sea level)", "Calcium Oxide", "Ferrous Oxide", cu2o)

    phase_fractions = [np.isclose(phantom, ii) for ii in range(int(phantom.max()) + 1)]

    phantom_obj = cct.physics.VolumeMaterial(phase_fractions, phase_compounds, voxel_size_cm)

    in_energy_keV = 17

    num_channels = len(elements) + (do_compton is True) + (do_attenuation is True)
    print("Producing signals (%d)" % num_channels)
    prjs_info = [None] * num_channels

    for ii, el in enumerate(elements):
        print("\nElement: '%s'" % el)
        _, phantom_yield = phantom_obj.get_fluo_yield(
            element=el, energy_in_keV=in_energy_keV, fluo_lines="KA", detector=detector
        )

        phantom_photons_expected = photon_density_per_pixel * phantom_yield
        buckets_yield = projector.fp(phantom_photons_expected)

        buckets_photons_measured, buckets_photons_expected, _ = cct.testing.add_noise(
            buckets_yield, 1, add_poisson=add_poisson, background_avg=background_avg
        )
        print("- Average photon counts per reading: %e" % np.mean(buckets_photons_expected))

        prjs_info[ii] = ProjectionData(
            signal_name="%s K-alpha" % el,
            photon_counts=buckets_photons_measured,
            num_incident_photons_beam=num_incident_photons_beam,
            total_measurement_time_s=total_measurement_time_s,
            photon_density_per_pixel=photon_density_per_pixel,
            incident_photons_pixel=incident_photons_pixel,
            expected_voxel_photon_counts=phantom_photons_expected,
            expected_photon_counts=buckets_photons_expected,
        )

    if do_compton:
        print("\nCompton")
        _, phantom_yield = phantom_obj.get_compton_scattering(energy_in_keV=in_energy_keV, detector=detector)

        phantom_photons_expected = photon_density_per_pixel * phantom_yield
        buckets_yield = projector.fp(phantom_photons_expected)

        buckets_photons_measured, buckets_photons_expected, _ = cct.testing.add_noise(
            buckets_yield, 1, add_poisson=add_poisson, background_avg=background_avg
        )
        print("- Average photon counts per reading: %e" % np.mean(buckets_photons_expected))

        prjs_info[ii + 1] = ProjectionData(
            signal_name="Compton",
            photon_counts=buckets_photons_measured,
            num_incident_photons_beam=num_incident_photons_beam,
            total_measurement_time_s=total_measurement_time_s,
            photon_density_per_pixel=photon_density_per_pixel,
            incident_photons_pixel=incident_photons_pixel,
            expected_voxel_photon_counts=phantom_photons_expected,
            expected_photon_counts=buckets_photons_expected,
        )
        ii += 1

    if do_attenuation:
        print("\nAttenuation")
        phantom_att = phantom_obj.get_attenuation(energy_keV=in_energy_keV)

        expected_beam_att = np.exp(-projector.fp(phantom_att))
        buckets_photons_measured, buckets_photons_expected, _ = cct.testing.add_noise(
            expected_beam_att, photon_density_per_pixel, add_poisson=add_poisson
        )
        print("- Average photon counts per reading: %e" % np.mean(buckets_photons_expected))
        observed_att = -np.log(buckets_photons_measured / photon_density_per_pixel)

        prjs_info[ii + 1] = ProjectionData(
            signal_name="Attenuation",
            photon_counts=observed_att,
            num_incident_photons_beam=num_incident_photons_beam,
            total_measurement_time_s=total_measurement_time_s,
            photon_density_per_pixel=photon_density_per_pixel,
            incident_photons_pixel=incident_photons_pixel,
            expected_voxel_photon_counts=phantom_att,
            expected_photon_counts=buckets_photons_measured,
        )

    return prjs_info
