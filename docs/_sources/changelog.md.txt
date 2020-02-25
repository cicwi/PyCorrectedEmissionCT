# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
## [Unreleased]
### Added
- <insert-features-not-in-a-release-yet>
### Fixed
- <insert-bugs-fixed-not-in-a-release-yet>
-->

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

[Unreleased]: https://www.github.com/cicwi/corrct/compare/v0.2.1...develop
