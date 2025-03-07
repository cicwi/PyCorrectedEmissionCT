# {py:mod}`corrct.physics.xraylib_helper`

```{py:module} corrct.physics.xraylib_helper
```

```{autodoc2-docstring} corrct.physics.xraylib_helper
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_element_number <corrct.physics.xraylib_helper.get_element_number>`
  - ```{autodoc2-docstring} corrct.physics.xraylib_helper.get_element_number
    :summary:
    ```
* - {py:obj}`get_element_number_and_symbol <corrct.physics.xraylib_helper.get_element_number_and_symbol>`
  - ```{autodoc2-docstring} corrct.physics.xraylib_helper.get_element_number_and_symbol
    :summary:
    ```
* - {py:obj}`get_compound <corrct.physics.xraylib_helper.get_compound>`
  - ```{autodoc2-docstring} corrct.physics.xraylib_helper.get_compound
    :summary:
    ```
* - {py:obj}`get_compound_cross_section <corrct.physics.xraylib_helper.get_compound_cross_section>`
  - ```{autodoc2-docstring} corrct.physics.xraylib_helper.get_compound_cross_section
    :summary:
    ```
````

### API

````{py:function} get_element_number(element: typing.Union[str, int]) -> int
:canonical: corrct.physics.xraylib_helper.get_element_number

```{autodoc2-docstring} corrct.physics.xraylib_helper.get_element_number
```
````

````{py:function} get_element_number_and_symbol(element: typing.Union[str, int]) -> tuple[str, int]
:canonical: corrct.physics.xraylib_helper.get_element_number_and_symbol

```{autodoc2-docstring} corrct.physics.xraylib_helper.get_element_number_and_symbol
```
````

````{py:function} get_compound(cmp_name: str, density: typing.Union[float, None] = None) -> dict
:canonical: corrct.physics.xraylib_helper.get_compound

```{autodoc2-docstring} corrct.physics.xraylib_helper.get_compound
```
````

````{py:function} get_compound_cross_section(compound: dict, mean_energy_keV: float) -> float
:canonical: corrct.physics.xraylib_helper.get_compound_cross_section

```{autodoc2-docstring} corrct.physics.xraylib_helper.get_compound_cross_section
```
````
