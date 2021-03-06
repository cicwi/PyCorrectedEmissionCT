# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]: https://www.github.com/cicwi/corrct/compare/v0.3.1...develop

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
