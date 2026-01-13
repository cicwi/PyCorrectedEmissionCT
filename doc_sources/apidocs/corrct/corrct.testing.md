# {py:mod}`corrct.testing`

```{py:module} corrct.testing
```

```{autodoc2-docstring} corrct.testing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`roundup_to_pow2 <corrct.testing.roundup_to_pow2>`
  - ```{autodoc2-docstring} corrct.testing.roundup_to_pow2
    :summary:
    ```
* - {py:obj}`download_phantom <corrct.testing.download_phantom>`
  - ```{autodoc2-docstring} corrct.testing.download_phantom
    :summary:
    ```
* - {py:obj}`create_phantom_nuclei3d <corrct.testing.create_phantom_nuclei3d>`
  - ```{autodoc2-docstring} corrct.testing.create_phantom_nuclei3d
    :summary:
    ```
* - {py:obj}`phantom_assign_concentration <corrct.testing.phantom_assign_concentration>`
  - ```{autodoc2-docstring} corrct.testing.phantom_assign_concentration
    :summary:
    ```
* - {py:obj}`phantom_assign_concentration_multi <corrct.testing.phantom_assign_concentration_multi>`
  - ```{autodoc2-docstring} corrct.testing.phantom_assign_concentration_multi
    :summary:
    ```
* - {py:obj}`add_noise <corrct.testing.add_noise>`
  - ```{autodoc2-docstring} corrct.testing.add_noise
    :summary:
    ```
* - {py:obj}`create_sino <corrct.testing.create_sino>`
  - ```{autodoc2-docstring} corrct.testing.create_sino
    :summary:
    ```
* - {py:obj}`create_sino_transmission <corrct.testing.create_sino_transmission>`
  - ```{autodoc2-docstring} corrct.testing.create_sino_transmission
    :summary:
    ```
* - {py:obj}`compute_error_power <corrct.testing.compute_error_power>`
  - ```{autodoc2-docstring} corrct.testing.compute_error_power
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NDArrayFloat <corrct.testing.NDArrayFloat>`
  - ```{autodoc2-docstring} corrct.testing.NDArrayFloat
    :summary:
    ```
````

### API

````{py:data} NDArrayFloat
:canonical: corrct.testing.NDArrayFloat
:value: >
   None

```{autodoc2-docstring} corrct.testing.NDArrayFloat
```

````

````{py:function} roundup_to_pow2(x: int | float | corrct.testing.NDArrayFloat, p: int, dtype: numpy.typing.DTypeLike = int) -> int | float | corrct.testing.NDArrayFloat
:canonical: corrct.testing.roundup_to_pow2

```{autodoc2-docstring} corrct.testing.roundup_to_pow2
```
````

````{py:function} download_phantom()
:canonical: corrct.testing.download_phantom

```{autodoc2-docstring} corrct.testing.download_phantom
```
````

````{py:function} create_phantom_nuclei3d(FoV_size: int | None = 100, dtype: numpy.typing.DTypeLike = np.float32) -> tuple[corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat]
:canonical: corrct.testing.create_phantom_nuclei3d

```{autodoc2-docstring} corrct.testing.create_phantom_nuclei3d
```
````

````{py:function} phantom_assign_concentration(ph_or: corrct.testing.NDArrayFloat, element: str = 'Ca', em_line: str = 'KA', in_energy_keV: float = 20.0, voxel_size_um: float = 0.5) -> tuple[corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat]
:canonical: corrct.testing.phantom_assign_concentration

```{autodoc2-docstring} corrct.testing.phantom_assign_concentration
```
````

````{py:function} phantom_assign_concentration_multi(ph_or: corrct.testing.NDArrayFloat, elements: collections.abc.Sequence[str] = ('Ca', 'Fe'), em_lines: str | collections.abc.Sequence[str] = 'KA', in_energy_keV: float = 20.0, detectors_pos_rad: float | None = None) -> tuple[list[corrct.testing.NDArrayFloat], corrct.testing.NDArrayFloat, list[corrct.testing.NDArrayFloat]]
:canonical: corrct.testing.phantom_assign_concentration_multi

```{autodoc2-docstring} corrct.testing.phantom_assign_concentration_multi
```
````

````{py:function} add_noise(img_clean: numpy.typing.NDArray, num_photons: int | float, add_poisson: bool = False, readout_noise_std: float | None = None, background_avg: float | None = None, background_std: float | None = None, detection_efficiency: float = 1.0, dtype: numpy.typing.DTypeLike = np.float32) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray, float]
:canonical: corrct.testing.add_noise

```{autodoc2-docstring} corrct.testing.add_noise
```
````

````{py:function} create_sino(ph: corrct.testing.NDArrayFloat, num_angles: int, start_angle_deg: float = 0.0, end_angle_deg: float = 180.0, dwell_time_s: float = 1.0, photon_flux: float = 1000000000.0, detectors_pos_rad: float | collections.abc.Sequence[float] | corrct.testing.NDArrayFloat = np.pi / 2, vol_att_in: corrct.testing.NDArrayFloat | None = None, vol_att_out: corrct.testing.NDArrayFloat | None = None, psf: corrct.testing.NDArrayFloat | None = None, background_avg: float | None = None, background_std: float | None = None, add_poisson: bool = False, readout_noise_std: float | None = None, dtype: numpy.typing.DTypeLike = np.float32) -> tuple[corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat, float]
:canonical: corrct.testing.create_sino

```{autodoc2-docstring} corrct.testing.create_sino
```
````

````{py:function} create_sino_transmission(ph: corrct.testing.NDArrayFloat, num_angles: int, start_angle_deg: float = 0, end_angle_deg: float = 180, dwell_time_s: float = 1, photon_flux: float = 1000000000.0, psf: corrct.testing.NDArrayFloat | None = None, add_poisson: bool = False, readout_noise_std: float | None = None, dtype: numpy.typing.DTypeLike = np.float32) -> tuple[corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat, corrct.testing.NDArrayFloat]
:canonical: corrct.testing.create_sino_transmission

```{autodoc2-docstring} corrct.testing.create_sino_transmission
```
````

````{py:function} compute_error_power(expected_vol: corrct.testing.NDArrayFloat, computed_vol: corrct.testing.NDArrayFloat) -> tuple[float, float]
:canonical: corrct.testing.compute_error_power

```{autodoc2-docstring} corrct.testing.compute_error_power
```
````
