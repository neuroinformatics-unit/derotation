[![Python Version](https://img.shields.io/pypi/pyversions/derotation.svg)](https://pypi.org/project/derotation)
[![PyPI Version](https://img.shields.io/pypi/v/derotation.svg)](https://pypi.org/project/derotation)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/neuroinformatics-unit/derotation/test_and_deploy.yml?label=CI)](https://github.com/neuroinformatics-unit/derotation/actions)
[![codecov](https://codecov.io/gh/neuroinformatics-unit/derotation/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/neuroinformatics-unit/derotation)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Derotation

The derotation package offers a robust solution for reconstructing images of rotating samples acquired with a line scanning microscope. It provides a set of tools to undo distortions caused by the rotation, including line-by-line derotation and center of rotation estimation.

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
To address certain neuroscience questions in rodents, it might be necessary to image the brain while the head or the body of the animal rotates. In such a case, and even more when the frame rate is low, the acquired movies are distorted by the rotation. These distortions have a peculiar pattern due to the line scanning nature of the microscope, which can be corrected by the derotation package. 

`derotation` provides a set of tools to undo these distortions:
- Recover calcium imaging movies by **line-by-line derotation** that can be fed into standard analysis pipelines such as suite2p;
- Estimate the **center of rotation** using ellipse fitting or Bayesian optimization;
- Validate improvements to the derotation algorithm and pipelines using synthetic data;
- Use debugging plots and logs to verify the quality of the derotation;
- Batch-process multiple datasets with consistent configuration files.

> [!Warning]
> 🏗️ The package is currently in early development and it requires rotation information coming from a step motor.

## Join the Development

Contributions to Derotation are encouraged, whether to fix a bug, develop a new feature, or improve documentation. Check out our [contributing guide](https://derotation.neuroinformatics.dev/community/contributing.html).

- [Open an issue](https://github.com/neuroinformatics-unit/derotation/issues) to report a bug or request a new feature.
- [Follow this Zulip topic](https://neuroinformatics.zulipchat.com/#narrow/channel/406001-Derotation/topic/Community.20Calls) to receive updates about upcoming Community Calls.

## Citation

If you use `derotation` in your work, please cite the following Zenodo DOI:

> Neuroinformatics Unit (2025). neuroinformatics-unit/derotation. Zenodo. [https://zenodo.org/doi/10.5281/zenodo.12755724](https://zenodo.org/doi/10.5281/zenodo.12755724)

## References
This package was built taking into account previous efforts on derotation algorithms, including:
- Voigts et al
- Hannestad et al

## License
⚖️ [BSD 3-Clause](./LICENSE)

## Package Template
This package layout and configuration (including pre-commit hooks and GitHub actions) have been copied from the [python-cookiecutter](https://github.com/neuroinformatics-unit/python-cookiecutter) template.

