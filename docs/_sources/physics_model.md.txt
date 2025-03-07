<!-- <script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script> -->

# Physics

The [`physics`](#corrct.physics) module includes functionality
for modeling and treating various signal related to X-ray physics.
In particular, we offer modeling of X-ray fluorescence (XRF) emission and X-ray
attenuation, and some useful tools for X-ray phase contrast.
The back-end of this module is the famous X-ray physics package called
[xraylib](https://github.com/tschoonj/xraylib).

## Attenuation

The [](#physics.attenuation) sub-module offers support for attenuation correction
in tomographic reconstructions, as well as some plotting functions of the different
attenuation effects.
For a dedicated walk-through on attenuation correction, we refer to the
[attenuation tutorial](attenuation_tutorial.md).

## X-ray Fluorescence

The [](#physics.xrf) sub-module offers support for working with X-ray fluorescence
(XRF), and contains mainly two classes: [](#physics.xrf.LinesSiegbahn) and
[](#physics.xrf.DetectorXRF).
The former exposes a simplified interface for handling XRF emission lines, using
the Siegbahn nomenclature. The latter allows one to describe the position and
geometry of a XRF detector, that is used in the `VolumeMaterial` class of the
[](#physics.materials) sub-module.

The module exposes two important functions: a static method of the
[](#physics.xrf.LinesSiegbahn) class, and a function:
```python
class LinesSiegbahn:
    """Siegbahn fluorescence lines collection class."""

    @staticmethod
    def get_lines(line: str) -> Sequence[FluoLine]:
        ...

def get_energy(
    element: Union[str, int],
    lines: Union[str, FluoLine, Sequence[FluoLine]],
    compute_average: bool = False,
    verbose: bool = False,
) -> Union[float, NDArray]:
    ...
```
The method [](#corrct.physics.xrf.LinesSiegbahn.get_lines) returns the list of
available lines for a given line family, e.g. the K<sub>&alpha;</sub> and
K<sub>&beta;</sub> lines for the K line family.
The function [](#corrct.physics.xrf.get_energy), instead, returns
the energy(ies) of the requested line(s) for a given element. If the requested
expression matches more than one line, it can either be the list of all the line
energies, or their average.

## Material modeling

The main class of the [](#physics.materials) sub-module is []`VolumeMaterial`, that allows one to model heterogeneous material
compositions in the reconstruction volume, with the aim of generating:
1. Attenuation maps (local linear attenuation coefficient).
2. Emission cross-sections maps for XRF and Compton.

## X-ray Phase Contrast

The [](#physics.phase) sub-module contains
functions to model the delta-over-beta value and transfer functions encountered
in phase contrast problems:
1. [](#physics.phase.get_delta_beta):
   Computes the delta-over-beta parameter for a specific compound given its molar composition, energy, and density.
2. [](#physics.phase.get_delta_beta_curves):
   Computes and optionally plots the delta-over-beta curves for a list of compounds over a specified energy range.
3. [](#physics.phase.plot_filter_responses):
   Plots the frequency response of the wave propagation for both TIE and CTF filters in either Fourier or direct space.
4. [](#physics.phase.get_propagation_filter):
   Computes the phase contrast propagation filter for given parameters, returning both Fourier-space and real-space filters.
5. [](#physics.phase.apply_propagation_filter):
   Applies a requested propagation filter (either TIE or CTF) to an image or stack of images.

## Unit conversion

The [](#physics.units) sub-module provides a small list of conversion functions
and classes to deal with conversions between different physical scales (e.g.
converting between `m` and `nm`) and different units of the electromagnetic
radiation (e.g. converting from energy to wavelength and vice versa).
In particular, here we find the classes [](#physics.units.ConversionMetric) and
[](#physics.units.ConversionEnergy), which provide the following functionality:

1. `ConversionMetric`: This class defines conversion factors between orders of
magnitude of metric units such as kilometers, meters, centimeters, etc.
It includes a `convert` method to convert numbers from a source unit to a destination unit.
2. `ConversionEnergy`: Similar to `ConversionMetric`, this class handles conversion
factors between orders of magnitude of energy units like GeV, MeV, keV, eV, etc.
It also provides a `convert` method for converting energy from a source unit to a destination unit.

The two functions [](#physics.units.energy_to_wlength) and
[](#physics.units.wlength_to_energy) convert energy to
wavelength and wavelength to energy, respectively.