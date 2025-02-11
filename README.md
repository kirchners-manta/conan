[![license](https://img.shields.io/badge/license-GPL%203.0-blue)](https://github.com/AK-Kirchner/conan_development/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-JCIM-green)](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01075)

# Confinement Analyzer


<img src="./docs/source/pictures/CONAN_logo.png" alt="CONAN Logo" width="500">


This is the GitHub repository containing the `CONAN` program developed by Prof. Kirchner's group at the University of Bonn.

## Installation

### For regular use
To install the package for regular use:
```bash
pip install .
```
### For development use
To install the development package
```bash
pip install -e '.[dev]'.
```
After installing the package, set the pre-commit hooks with the following command
```bash
pre-commit install
```
This ensures that the code is formatted according to the
PEP 8 guidelines.

You can test your current code with
```bash
 pre-commit run --all-files
```

## Usage
After installing the package, conan is available via the command line:
```bash
CONAN -h
```

## Documentation

Read the manual at [read-the-docs](https://con-an.readthedocs.io), or generate the documentation locally with:
```bash
tox -e docs
```
The built documentation can be found here: `docs/build/html/index.html`.

## Citations
Please cite the publication, if you use the program in your research.

https://pubs.acs.org/doi/10.1021/acs.jcim.3c01075
https://doi.org/10.1021/acs.jpcb.3c08493
