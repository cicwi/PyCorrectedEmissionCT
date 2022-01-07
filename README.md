# PyCorrectedEmissionCT

[![Python package](https://github.com/cicwi/PyCorrectedEmissionCT/actions/workflows/pythonpackage.yml/badge.svg)](https://github.com/cicwi/PyCorrectedEmissionCT/actions/workflows/pythonpackage.yml)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/cicwi/PyCorrectedEmissionCT)
![License](https://img.shields.io/github/license/cicwi/PyCorrectedEmissionCT)
[![DOI](https://zenodo.org/badge/218092017.svg)](https://zenodo.org/badge/latestdoi/218092017)

Physically corrected projectors for X-ray induced emission CT.

This package provides the following functionality:

* Support for correction of the forward-projection and back-projection.
* Includes the following solvers (reconstruction algorithms):
  - Simultaneous Iterative Reconstruction Technique (SIRT).
  - Simultaneous Algebraic Reconstruction Technique (SART).
  - Primal-dual optimization from Chambolle-Pock, with:
    * Various data fitting terms, including Gaussian and Poisson noise modelling.
    * Various optional regularization terms, including: TV-min, l1-min, laplacian, and wavelet l1-min.

It contains the code used for the following paper, which also provides a
mathematical description of the concepts and algorithms used here:

* N. Viganò and V. A. Solé, "Physically corrected forward operators for
induced emission tomography: a simulation study," Meas. Sci. Technol., no.
Advanced X-Ray Tomography, pp. 1–26, Nov. 2017.  
[https://doi.org/10.1088/1361-6501/aa9d54](https://doi.org/10.1088/1361-6501/aa9d54)

Other useful information:

* Free software: BSD 3-Clause license
* Documentation: [https://cicwi.github.io/PyCorrectedEmissionCT/](https://cicwi.github.io/PyCorrectedEmissionCT/)

## Getting Started

It takes a few steps to setup PyCorrectedEmissionCT on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.

### Installing with conda

Simply install with:
```
conda install -c n-vigano corrct
```

If you want fast tomographic projectors using the astra-toolbox:
```
conda install -c astra-toolbox astra-toolbox
```

### Installing from PyPI

Simply install with:
```
pip install corrct
```

If you are on jupyter, and don't have the rights to install packages
system-wide (e.g. on jupyter-slurm at ESRF), then you can install with:
```
! pip install --user corrct
```

### Installing from source

To install PyCorrectedEmissionCT, simply clone this GitHub
project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/cicwi/PyCorrectedEmissionCT.git corrct
cd corrct
pip install -e .
```

### Running the examples

To learn more about the functionality of the package check out our
examples folder.

## Authors and contributors

* **Nicola VIGANÒ** - *Initial work*

See also the list of [contributors](https://github.com/cicwi/PyCorrectedEmissionCT/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the BSD license - see the [LICENSE.md](LICENSE.md) file for details.
