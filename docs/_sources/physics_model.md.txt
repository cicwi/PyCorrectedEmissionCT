<!-- <script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script> -->

# Physics

The [`physics`](corrct.html#module-corrct.physics) module includes functionality
for modeling and treating various signal related to X-ray physics.
In particular, we offer modeling of X-ray fluorescence (XRF) emission and X-ray
attenuation, and some useful tools for X-ray phase contrast.
The back-end of this module is the famous X-ray physics package called
[xraylib](https://github.com/tschoonj/xraylib).

## X-ray Fluorescence

The [`physics.xrf`](corrct.html#module-corrct.physics.xrf) sub-module offers
mainly two classes: `LinesSiegbahn` and `DetectorXRF`.
The former exposes a simplified interface for handling XRF emission lines, using
the Siegbahn nomenclature. The latter allows one to describe the position and
geometry of a XRF detector, that is used in the `VolumeMaterial` class of the
[`physics.materials`](corrct.html#module-corrct.physics.materials) sub-module.

The `LinesSiegbahn` class, exposes two important static methods:
```python
class LinesSiegbahn:
    """Siegbahn fluorescence lines collection class."""

    @staticmethod
    def get_lines(line: str) -> Sequence[FluoLine]:
        ...

    @staticmethod
    def get_energy(
        element: Union[str, int],
        lines: Union[str, FluoLine, Sequence[FluoLine]],
        compute_average: bool = False,
        verbose: bool = False,
    ) -> Union[float, NDArray]:
        ...
```
The method `get_lines` returns the list of available lines for a given line
family, e.g. the K<sub>&alpha;</sub> and K<sub>&beta;</sub> lines for the K line family.
The method `get_energy`, instead, returns the energy(ies) of the requested line(s)
for a given element. If the requested expression matches more than one line, it
can either be the list of all the line energies, or their average.

## Material modeling

The main class of the [`physics.materials`](corrct.html#module-corrct.physics.materials)
sub-module is `VolumeMaterial`, that allows one to model heterogeneous material
compositions in the reconstruction volume, with the aim of generating:
1. Attenuation maps (local linear attenuation coefficient).
2. Emission cross-sections maps for XRF and Compton.

## X-ray Phase Contrast

The [`physics.phase`](corrct.html#module-corrct.physics.phase) sub-module contains
functions to model the delta-over-beta value and transfer functions encountered
in phase contrast problems.
