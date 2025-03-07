# {py:mod}`corrct.physics`

```{py:module} corrct.physics
```

```{autodoc2-docstring} corrct.physics
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

corrct.physics.materials
corrct.physics.phase
corrct.physics.xraylib_helper
corrct.physics.xrf
corrct.physics.attenuation
corrct.physics.units
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`pencil_beam_profile <corrct.physics.pencil_beam_profile>`
  - ```{autodoc2-docstring} corrct.physics.pencil_beam_profile
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__author__ <corrct.physics.__author__>`
  - ```{autodoc2-docstring} corrct.physics.__author__
    :summary:
    ```
* - {py:obj}`__email__ <corrct.physics.__email__>`
  - ```{autodoc2-docstring} corrct.physics.__email__
    :summary:
    ```
* - {py:obj}`xraylib <corrct.physics.xraylib>`
  - ```{autodoc2-docstring} corrct.physics.xraylib
    :summary:
    ```
* - {py:obj}`get_compound <corrct.physics.get_compound>`
  - ```{autodoc2-docstring} corrct.physics.get_compound
    :summary:
    ```
* - {py:obj}`get_element_number <corrct.physics.get_element_number>`
  - ```{autodoc2-docstring} corrct.physics.get_element_number
    :summary:
    ```
* - {py:obj}`FluoLinesSiegbahn <corrct.physics.FluoLinesSiegbahn>`
  - ```{autodoc2-docstring} corrct.physics.FluoLinesSiegbahn
    :summary:
    ```
* - {py:obj}`VolumeMaterial <corrct.physics.VolumeMaterial>`
  - ```{autodoc2-docstring} corrct.physics.VolumeMaterial
    :summary:
    ```
* - {py:obj}`convert_energy_to_wlength <corrct.physics.convert_energy_to_wlength>`
  - ```{autodoc2-docstring} corrct.physics.convert_energy_to_wlength
    :summary:
    ```
* - {py:obj}`convert_wlength_to_energy <corrct.physics.convert_wlength_to_energy>`
  - ```{autodoc2-docstring} corrct.physics.convert_wlength_to_energy
    :summary:
    ```
````

### API

````{py:data} __author__
:canonical: corrct.physics.__author__
:value: >
   'Nicola VIGANÒ'

```{autodoc2-docstring} corrct.physics.__author__
```

````

````{py:data} __email__
:canonical: corrct.physics.__email__
:value: >
   'N.R.Vigano@cwi.nl'

```{autodoc2-docstring} corrct.physics.__email__
```

````

````{py:data} xraylib
:canonical: corrct.physics.xraylib
:value: >
   None

```{autodoc2-docstring} corrct.physics.xraylib
```

````

````{py:data} get_compound
:canonical: corrct.physics.get_compound
:value: >
   None

```{autodoc2-docstring} corrct.physics.get_compound
```

````

````{py:data} get_element_number
:canonical: corrct.physics.get_element_number
:value: >
   None

```{autodoc2-docstring} corrct.physics.get_element_number
```

````

````{py:data} FluoLinesSiegbahn
:canonical: corrct.physics.FluoLinesSiegbahn
:value: >
   None

```{autodoc2-docstring} corrct.physics.FluoLinesSiegbahn
```

````

````{py:data} VolumeMaterial
:canonical: corrct.physics.VolumeMaterial
:value: >
   None

```{autodoc2-docstring} corrct.physics.VolumeMaterial
```

````

````{py:data} convert_energy_to_wlength
:canonical: corrct.physics.convert_energy_to_wlength
:value: >
   None

```{autodoc2-docstring} corrct.physics.convert_energy_to_wlength
```

````

````{py:data} convert_wlength_to_energy
:canonical: corrct.physics.convert_wlength_to_energy
:value: >
   None

```{autodoc2-docstring} corrct.physics.convert_wlength_to_energy
```

````

````{py:function} pencil_beam_profile(voxel_size_um: float, beam_fwhm_um: float, profile_size: int = 1, beam_shape: str = 'gaussian', verbose: bool = False) -> numpy.typing.NDArray
:canonical: corrct.physics.pencil_beam_profile

```{autodoc2-docstring} corrct.physics.pencil_beam_profile
```
````
