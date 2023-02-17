# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]: https://www.github.com/cicwi/corrct/compare/v0.7.2...develop

## 0.7.2 - 2023-02-17
### Added
- Default VolumeGeometry creation from volume.
- Tapering border size support in volume mask.
- 1D gradient based regularizer shortcut.
- Isotropic undecimented wavelet regularizer support.
### Fixed
- Fluorescence line selection case sensitive bug.
- Verbose feedback in parabolic fitting for parameter tuning.

## 0.7.1 - 2022-10-31
### Added
- More documentation / tutorial content.
- Energy conversion functions (keV to m, and viceversa).
- Tapering for truncated volumes in FSC computation.
### Fixed
- Typo that preventing using `ProjectionGeometry` from `models`.
- Shape dimensions (XYZ vs ZXY) interpretation in `VolumeGeometry` and other functions.
- Cross-validation data term initialization in solvers.
- Tapering to circular masks.

## 0.7.0 - 2022-10-11
### Added
- Wide API refactoring: renamed various modules, and moved processing routines to sub-package.
- Initial draft of tutorial. More documentation on the way.
- Line intersection computation.
- Plotting of FRCs.
- Tests: initial support for projectors.
### Fixed
- Typo in function name.
- Simplified examples 01 and 02.
- FRC/FSC axes selection.

## 0.6.0 - 2022-09-08
### Added
- Multi-channel (collaborative) regularization (including TNV - Total Nuclear Variation).
- Support for custom (external, user defined) projection backends.
- Dedicated FBP filters module.
- Parallelization of guided regularization functions.
- A lot of work towards full type hinting usage.
### Fixed
- Potential memory leak in astra-toolbox projector backend.
- Gradient and laplacian implementations for arbitrary directions.
- Output and consistency from tests for solver classes.
- Residual norm computation in the weighted least squares data term.
- Dimension mismatch in projector, when using only one angle.
- Back-projection normalization in scikit-image projector backend.

## 0.5.1 - 2022-06-23
### Added
- Better support for FBP:
  - Dedicated solver class.
  - Support for 3D geometries.
  - Use of scikit-image's filters, with `rfft`.
  - Support for various padding modes (constant, edge, etc).
- Support for various padding modes in gradient, Laplacian and convolution operators.
- Support for Lorentzian and sech^2 beam shapes.
- Computation of Center-of-Mass for volume.
### Fixed
- Border behavior in convolution operator.
- Sanitized projector's angle input.
- Denoising / deconvolution method.
- Models: handling of astra-toolbox's 2D rotation direction.

## 0.5.0 - 2022-04-14
### Added
- New module called `models`:
  - VolumeGeometry: description of the volume.
  - ProjectionGeometry: description / manipulation of the projection geometry.
  - Support for unmatched pixel-voxel size, cone beam geometry, uneven volume shape.
  - Arbitrary geometry, including tilted detector.
- Convolution operator and deconvolution support.
- Support for rotating projection stack.
- Support for computing cross-correlation curves.
- Computation of PSF for pencil beam scanning with gaussian beam.
- Support for computing attenuation maps outside the corrected projector.
### Fixed
- Pixel weights, when having small outliers.
- Volume mask generation
- Dimension bug in azimuthal integration.
- Inconsistency between astra-toolbox and scikit-image backends.

## 0.4.2 - 2022-01-14
### Added
- Support for XRF line handling (Siegbahn).
- Support for log-scaling of data weigths.
- More type hinting and docstrings.
### Fixed
- 3D volume support (especially in astra-toolbox backend).
- Attenuation correction in 3D volumes.
- Handling of multi-detector in attenuation corrected projector (non-symmetric).
- Handling of multi-detector in SART.
- Wavelet scaling in regularizers.
- Installation instructions.
- Flat-fielding function.
- Updated examples to latest solvers changes.

## 0.4.1 - 2021-12-06
### Added
- Uncertainty propagation functions, for weighted minimizations.
- Unpreconditioned l1-minimization (standard behavior).
### Fixed
- Double application of weights in SIRT weighted least squares.
- Use PDHG preconditioning by default.
- Wavelet approximation minimization by default.

## 0.4.0 - 2021-11-29
### Added
- Moved data terms and regularizers to dedicated module.
- Conditional minimization of low pass (approximation) portion of wavelet decomposition.
- Cross-validation and L-curve methods for selecting regularization parameters.
- Support for different regularization weights across reconstruction volume.
- Improved corrected projector performance.
- Restructured and expanded test phantom creation.
- Basic multi-threading support for CPU operations.
- Initial type hinting.
- FRC/FSC and azimuthal integration support.
- Automatic image denoising (with cross-validation).
### Fixed
- Size of transformed volumes in dwt.
- Deprecation warnings from recent numpy versions.
- Warning in scikit-image projector backend.
- Copy of x0 and data terms in solvers.
- Angle consistency among different projectors.
- ProjectorMatrix when using precondioned PDHG solver.

## 0.3.2 - 2021-04-15
### Added
- scikit-image backend, as preferred in 2D non-GPU reconstructions.
- Renamed AttenuationProjector into ProjectorAttenuationXRF.
- Support for test/cross-validation sets in solvers.
- Residual calculation with different data-fidelity terms.
- Projection matrix based projector.
### Fixed
- PyWavelets interface change.
- Background usage in the data terms.
- Examples on the different data-fidelity terms.
- Github workflows to use pip instead of conda.

## 0.3.1 - 2020-10-20
### Added
- FFT operator, and FFT regularizer.
- l2-gradient (smooth) regularizer.
- Support for multiple regularizers.
### Fixed
- Norm l2b (deadzone) is also weighted now.
- Weighted least-squares implementation.
- Crop inverse DWT output when necessary.
- Changelog of 0.3.0

## 0.3.0 - 2020-09-23
### Added
- Data fidelity classes:
  - Use as norms for regularizers.
  - New classes: Huber norm, l12, l2b (l2 with deadzone), l1b (l1 with deadzone).
  - Residual computation.
  - Background bias support.
- Regularization:
  - New classes: smoothness based on gradient, Huber TV, decimated wavelets, Huber wavelet, median filter.
  - Constraint classes (lower and upper limits) based on regularization.
  - Simplified and unified code.
- Solvers:
  - SIRT now supports various l2 norm data terms.
  - CP and SIRT report better information on regularization.
  - Use of data term to compute residual correctly.
- More flexibility and options to image denoiser.
- New module for easing development of tests and examples.
### Fixed
- Python hard crash when CUDA not available.
- Examples to comply with recent changes.
- Axes passing to wavelet transform.
- Stationary wavelet weights calculation.

## 0.2.4 - 2020-06-01
### Added
- Operator form (based on scipy.linalg.LinearOperator) for regularizers and projectors
- Data fidelity function classes, including weighted least-squares
- More test coverage
- Support for non GPU reconstructions (in 2D)
- Multi-dimensional volume mask creation
### Fixed
- SIRT regularization
- Windows conda package build, and Python 3.8 target
- Laplacian regularizer norm
- Wavelet regularizer normalization

## 0.2.3 - 2020-05-05
### Added
- On-demand padding for Wavelet regularizer
- Projection intensity rescaling for different angles in base projector
- Support for 3D absorption maps (but not thouroughly tested, yet)
- Data-term residual computation to all solvers
- Some tests to solvers and regularizers
- Automated testing and linting on github workflows
- Support for ASTRA's super-sampling of pixels and voxels
- Implemented scipy's sparse LinearOperator interface for the projectors
- Utility min-log and simple flat-fielding functions for transmission data
### Fixed
- Copy-paste error, and detector data axis order in SART algorithm implementation
- Weights for masks in SART algorithm
- Error in unpreconditioned CP implementation
- Applied linting changes, to improve the readability of the code

## 0.2.2 - 2020-03-30
### Added
- Unpreconditioned Chambolle-Pock algorithm (as default)
- Utility functions for sinogram padding and circular volume mask creation
- FBP data-dependent filters from D. Pelt
- Improved performance of backprojection in non-symmetric uncorrected
### Fixed
- Wavelet decomposition along one dimension
- TV regularizer for dimensions larger than ndims
- Documentation and links
- Handling of matrices

## 0.2.1 - 2020-02-25
### Added
- New regularizers: non-decimated wavelets and laplacian
- Solvers: added projection masks, relaxation parameters, sirt regularizers
- Uncorrected projector for 2D and 3D data
### Fixed
- sub-pixel attenuation correction

## 0.2.0 - 2019-11-05
### Added
- Multi-detector reconstruction
### Fixed
- TV-min sign
- Solvers' name visualization
- Back-projection behavior with different inputs

## 0.1.0 - 2019-10-29
### Added
- Initial release, with corrected forward and back-projections.
- SIRT, SART, and Chambolle-Pock solvers.
- TV-min and l1-norm based regularizers.
