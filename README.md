# CONAN
This GitHub repository contains the CONAN program.

Please find the corresponding publication under the following link:
https://pubs.acs.org/doi/10.1021/acs.jcim.3c01075

The program and manual is maintained by Leonard Dick (dick@thch.uni-bonn.de).
Please cite the publication, if you use the program in your research.

Read the manual here:
https://con-an.readthedocs.io

## Install
### For Development Usage
To install the package for development:
```bash
pip install -e '.[dev]'
```
After installing the package, set up the pre-commit hooks with the following command:
```bash
pre-commit install
```
This ensures that contributors only commit code formatted according to
PEP 8 guidelines.

You can test your current code using:
```bash
 pre-commit run --all-files
```

### For Regular Usage:
To install the package for regular usage:
```bash
pip install .
```

## Usage
After installing the package, conan is available via the command line:
```bash
CONAN -h
```

## Build Documentation
To generate the documentation locally, run:
```bash
tox -e docs
```
The built documentation can be found here: docs/build/html/index.html.
