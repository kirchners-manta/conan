# CONAN
This GitHub repository contains the CONAN program.

Please find the corresponding publication under the following link:
https://pubs.acs.org/doi/10.1021/acs.jcim.3c01075

The program and manual is maintained by Leonard Dick (dick@thch.uni-bonn.de).
Please cite the publication, if you use the program in your research.

Read the manual here:
https://con-an.readthedocs.io

## Install
### For development use:
```bash
pip install -e '.[dev]'
```
After installing the package use the following command to setup the pre-commit hooks.
```bash
pre-commit install
```
This will make sure that contributor will only commit code that is formatted according to
pep 8 code style guideline.

You can test your current code using the following command
```bash
 pre-commit run --all-files
```

### For usage:

```bash
pip install .
```

## Usage
After installing the package conan is available via the command line.
```bash
CONAN -h
```
