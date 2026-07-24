# {py:mod}`corrct.alignment.markers`

```{py:module} corrct.alignment.markers
```

```{autodoc2-docstring} corrct.alignment.markers
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MarkerTrackingVisualizer <corrct.alignment.markers.MarkerTrackingVisualizer>`
  - ```{autodoc2-docstring} corrct.alignment.markers.MarkerTrackingVisualizer
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`cm2inch <corrct.alignment.markers.cm2inch>`
  - ```{autodoc2-docstring} corrct.alignment.markers.cm2inch
    :summary:
    ```
* - {py:obj}`track_marker <corrct.alignment.markers.track_marker>`
  - ```{autodoc2-docstring} corrct.alignment.markers.track_marker
    :summary:
    ```
* - {py:obj}`create_marker_disk <corrct.alignment.markers.create_marker_disk>`
  - ```{autodoc2-docstring} corrct.alignment.markers.create_marker_disk
    :summary:
    ```
````

### API

````{py:function} cm2inch(dims: collections.abc.Sequence[float] | numpy.typing.NDArray) -> tuple[float]
:canonical: corrct.alignment.markers.cm2inch

```{autodoc2-docstring} corrct.alignment.markers.cm2inch
```
````

````{py:function} track_marker(prj_data: numpy.typing.NDArray, marker_vu: numpy.typing.NDArray, stack_axis: int = -2) -> numpy.typing.NDArray
:canonical: corrct.alignment.markers.track_marker

```{autodoc2-docstring} corrct.alignment.markers.track_marker
```
````

````{py:function} create_marker_disk(data_shape_vu: collections.abc.Sequence[int] | numpy.typing.NDArray, radius: float, super_sampling: int = 5, conv: bool = True) -> numpy.typing.NDArray
:canonical: corrct.alignment.markers.create_marker_disk

```{autodoc2-docstring} corrct.alignment.markers.create_marker_disk
```
````

`````{py:class} MarkerTrackingVisualizer(fitted_positions_vu: numpy.typing.NDArray, images: numpy.typing.NDArray, marker: numpy.typing.NDArray, trajectory: corrct.alignment.fitting.Trajectory | None = None)
:canonical: corrct.alignment.markers.MarkerTrackingVisualizer

```{autodoc2-docstring} corrct.alignment.markers.MarkerTrackingVisualizer
```

```{rubric} Initialization
```

```{autodoc2-docstring} corrct.alignment.markers.MarkerTrackingVisualizer.__init__
```

````{py:method} _update() -> None
:canonical: corrct.alignment.markers.MarkerTrackingVisualizer._update

```{autodoc2-docstring} corrct.alignment.markers.MarkerTrackingVisualizer._update
```

````

````{py:method} _key_event(evnt) -> None
:canonical: corrct.alignment.markers.MarkerTrackingVisualizer._key_event

```{autodoc2-docstring} corrct.alignment.markers.MarkerTrackingVisualizer._key_event
```

````

````{py:method} _scroll_event(evnt) -> None
:canonical: corrct.alignment.markers.MarkerTrackingVisualizer._scroll_event

```{autodoc2-docstring} corrct.alignment.markers.MarkerTrackingVisualizer._scroll_event
```

````

`````
