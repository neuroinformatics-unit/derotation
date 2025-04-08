[![Python Version](https://img.shields.io/pypi/pyversions/derotation.svg)](https://pypi.org/project/derotation)
[![PyPI Version](https://img.shields.io/pypi/v/derotation.svg)](https://pypi.org/project/derotation)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://img.shields.io/github/actions/workflow/status/neuroinformatics-unit/derotation/test_and_deploy.yml?label=CI)](https://github.com/neuroinformatics-unit/derotation/actions)
[![codecov](https://codecov.io/gh/neuroinformatics-unit/derotation/branch/main/graph/badge.svg?token=P8CCH3TI8K)](https://codecov.io/gh/neuroinformatics-unit/derotation)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/format.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# Derotation

The derotation package offers a robust solution for reconstructing movies of rotating samples acquired with a line scanning microscope.

![](docs/source/_static/dark_derotation.gif)

## Quick Install

Install the latest stable release in a conda environment:
```bash
conda create -n derotation-env python=3.12
conda activate derotation-env
pip install derotation
```

> [!Note]
> Read the [documentation](https://derotation.neuroinformatics.dev) for more information, including [examples](https://derotation.neuroinformatics.dev/examples/index.html) and [API reference](https://derotation.neuroinformatics.dev/api_index.html).

## Overview
To address certain neuroscience questions, it might be necessary to image the brain while the head or the body of the animal rotates. In such a case, and even more when the frame rate is low, the acquired movies are distorted by the rotation. These distortions have a peculiar pattern due to the line scanning nature of the microscope, which can be corrected by the derotation package.

![](_static/dark_derotation.gif)

With `derotation` you can:
- Recover calcium imaging movies by **line-by-line derotation** that can be fed into standard analysis pipelines such as [suite2p](https://github.com/MouseLand/suite2p)
- Estimate the **center of rotation** using Bayesian optimization
- Validate improvements to the derotation algorithm and pipelines using synthetic data
- Verify the quality of the derotation using debugging plotting tools
- Batch-process multiple datasets with consistent configuration files

> [!Warning]
> 🏗️ The package is currently in early development and it requires rotation information coming from a step motor.

## Data Source & Funding
All microscopy data presented here as an example has been acquired with a 3-photon microscope by [Simon Weiler](https://github.com/simonweiler) in the [Margrie Lab](https://www.sainsburywellcome.org/web/groups/margrie-lab).

This project was sponsored by the [Margrie Lab](https://www.sainsburywellcome.org/web/groups/margrie-lab) in the [Sainsbury Wellcome Centre for Neural Circuits and Behaviour](https://www.sainsburywellcome.org/web/) at University College London.

## References
This package was inspired by [previous work on derotation](https://github.com/jvoigts/rotating-2p-image-correction) shown in [Voigts & Harnett, 2020, Neuron](https://doi.org/10.1016/j.neuron.2019.10.016).


## Join the Development

Contributions to `derotation` are encouraged, whether to fix a bug, develop a new feature, or improve documentation. Get in touch through our [Zulip chat](https://neuroinformatics.zulipchat.com/#narrow/channel/495735-Derotation).

[Open an issue](https://github.com/neuroinformatics-unit/derotation/issues) to report a bug or request a new feature.
