[![Python Version](https://img.shields.io/pypi/pyversions/derotation.svg)](https://pypi.org/project/derotation)
[![PyPI Version](https://img.shields.io/pypi/v/derotation.svg)](https://pypi.org/project/derotation)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/neuroinformatics-unit/derotation/test_and_deploy.yml?label=CI)](https://github.com/neuroinformatics-unit/derotation/actions)
[![codecov](https://codecov.io/gh/neuroinformatics-unit/derotation/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/neuroinformatics-unit/derotation)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Derotation

The derotation package offers a robust solution for reconstructing images of rotating samples acquired with a line scanning microscope. Although it's tailored for calcium imaging data, this versatile tool is adaptable for any dataset where rotation angles are recorded, making it an essential utility for a wide array of image processing applications.

The core algorithm, `rotate_an_image_array_line_by_line`, can also be used as a standalone function to deform images by a given angle.

![](docs/source/_static/derotation_overview.png)

## Quick Install

To install the latest stable release:
```bash
pip install derotation
```

For a conda environment:
```bash
conda create -n derotation-env python=3.12
conda activate derotation-env
```

> [!Note]
> Read the [documentation](https://derotation.neuroinformatics.dev) for more information, including [full installation instructions](https://derotation.neuroinformatics.dev/user_guide/installation.html) and [examples](https://derotation.neuroinformatics.dev/examples/index.html).

## Overview

Derotation is designed for reconstructing image stacks affected by controlled rotations delivered via a motor. It provides tools for estimating the center of rotation, optimizing alignment, and batch processing across multiple datasets. 

- **Flexible Derotation**: Supports any rotation protocol, including continuous and discontinuous rotations.
- **Optimized Center of Rotation Estimation**: Uses Bayesian Optimization or ellipse fitting.
- **Batch Processing**: Automates derotation for multiple datasets.
- **Simulated Data**: Generates synthetic datasets for testing and validation.
- **Comprehensive Debugging Plots**: Visualize the derotation process at each step.

Find out more in our [mission and scope](https://derotation.neuroinformatics.dev/community/mission-scope.html) statement and our [roadmap](https://derotation.neuroinformatics.dev/community/roadmaps.html).

## Limitations

> [!Warning]
> 🏗️ The package is currently in early development and the interface is subject to change. Feel free to play around and provide feedback.

The current version of Derotation has the following limitations:
- It expects the image stack to be a TIFF file.
- It expects to receive four analog signal inputs: 
  - rotation on signal
  - line clock
  - frame clock
  - rotation ticks (expects a step motor signal)
  
  Rotation ticks are used to determine the rotation angle at each frame.

## Join the Development

Contributions to Derotation are encouraged, whether to fix a bug, develop a new feature, or improve documentation. Check out our [contributing guide](https://derotation.neuroinformatics.dev/community/contributing.html).

- [Open an issue](https://github.com/neuroinformatics-unit/derotation/issues) to report a bug or request a new feature.
- [Follow this Zulip topic](https://neuroinformatics.zulipchat.com/#narrow/channel/406001-Derotation/topic/Community.20Calls) to receive updates about upcoming Community Calls.

## Citation

If you use Derotation in your work, please cite the following Zenodo DOI:

> Neuroinformatics Unit (2025). neuroinformatics-unit/derotation. Zenodo. [https://zenodo.org/doi/10.5281/zenodo.12755724](https://zenodo.org/doi/10.5281/zenodo.12755724)

## License
⚖️ [BSD 3-Clause](./LICENSE)

## Package Template
This package layout and configuration (including pre-commit hooks and GitHub actions) have been copied from the [python-cookiecutter](https://github.com/neuroinformatics-unit/python-cookiecutter) template.

