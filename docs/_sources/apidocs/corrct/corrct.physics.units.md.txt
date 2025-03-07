# {py:mod}`corrct.physics.units`

```{py:module} corrct.physics.units
```

```{autodoc2-docstring} corrct.physics.units
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ConversionMetric <corrct.physics.units.ConversionMetric>`
  - ```{autodoc2-docstring} corrct.physics.units.ConversionMetric
    :summary:
    ```
* - {py:obj}`ConversionEnergy <corrct.physics.units.ConversionEnergy>`
  - ```{autodoc2-docstring} corrct.physics.units.ConversionEnergy
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`energy_to_wlength <corrct.physics.units.energy_to_wlength>`
  - ```{autodoc2-docstring} corrct.physics.units.energy_to_wlength
    :summary:
    ```
* - {py:obj}`wlength_to_energy <corrct.physics.units.wlength_to_energy>`
  - ```{autodoc2-docstring} corrct.physics.units.wlength_to_energy
    :summary:
    ```
````

### API

`````{py:class} ConversionMetric
:canonical: corrct.physics.units.ConversionMetric

```{autodoc2-docstring} corrct.physics.units.ConversionMetric
```

````{py:attribute} str_to_order
:canonical: corrct.physics.units.ConversionMetric.str_to_order
:value: >
   None

```{autodoc2-docstring} corrct.physics.units.ConversionMetric.str_to_order
```

````

````{py:attribute} order_to_str
:canonical: corrct.physics.units.ConversionMetric.order_to_str
:value: >
   None

```{autodoc2-docstring} corrct.physics.units.ConversionMetric.order_to_str
```

````

````{py:method} convert(from_unit: str, to_unit: str) -> float
:canonical: corrct.physics.units.ConversionMetric.convert
:staticmethod:

```{autodoc2-docstring} corrct.physics.units.ConversionMetric.convert
```

````

`````

`````{py:class} ConversionEnergy
:canonical: corrct.physics.units.ConversionEnergy

```{autodoc2-docstring} corrct.physics.units.ConversionEnergy
```

````{py:attribute} str_to_order
:canonical: corrct.physics.units.ConversionEnergy.str_to_order
:value: >
   None

```{autodoc2-docstring} corrct.physics.units.ConversionEnergy.str_to_order
```

````

````{py:attribute} order_to_str
:canonical: corrct.physics.units.ConversionEnergy.order_to_str
:value: >
   None

```{autodoc2-docstring} corrct.physics.units.ConversionEnergy.order_to_str
```

````

````{py:method} convert(from_unit: str, to_unit: str) -> float
:canonical: corrct.physics.units.ConversionEnergy.convert
:staticmethod:

```{autodoc2-docstring} corrct.physics.units.ConversionEnergy.convert
```

````

`````

````{py:function} energy_to_wlength(energy: typing.Union[float, numpy.typing.NDArray], unit_wl: str = 'm', unit_en: str = 'keV') -> typing.Union[float, numpy.typing.NDArray]
:canonical: corrct.physics.units.energy_to_wlength

```{autodoc2-docstring} corrct.physics.units.energy_to_wlength
```
````

````{py:function} wlength_to_energy(w_length: typing.Union[float, numpy.typing.NDArray], unit_wl: str = 'm', unit_en: str = 'keV') -> typing.Union[float, numpy.typing.NDArray]
:canonical: corrct.physics.units.wlength_to_energy

```{autodoc2-docstring} corrct.physics.units.wlength_to_energy
```
````
