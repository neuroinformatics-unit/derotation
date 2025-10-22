---
title: 'derotation: A Python package for correcting rotation-induced distortions in line-scanning microscopy'
tags:
- Python
- neuroscience
- calcium imaging
- two-photon microscopy
- motion correction
- image processing
- line-scanning
authors:
  - name: Laura Porta
    affiliation: "1"
affiliations:
  - name: Sainsbury Wellcome Centre for Neural Circuits and Behaviour, University College London, UK
    index: 1
date: 2025-08-12
bibliography: paper.bib
---

# Summary

Line-scanning microscopy, including multi-photon calcium imaging, is a powerful technique for observing dynamic processes at cellular resolution. However, when the imaged sample rotates during acquisition, the sequential line-by-line scanning process introduces geometric distortions. These artifacts, which manifest as shearing or curving of features, can severely compromise downstream analyses such as motion registration, cell detection, and signal extraction. While several studies have developed custom solutions for this issue [1-4], a general-purpose, accessible software package has been lacking.

derotation is an open-source Python package that algorithmically reconstructs undistorted movies from data acquired during sample rotation. By synchronizing recorded rotation angles with the microscope's line acquisition clock, the software applies a precise, line-by-line inverse transformation to restore the original geometry of the imaged plane. This correction enables reliable quantitative imaging during rapid rotational movements, making it possible to study yaw motion without sacrificing image quality.

![Example of derotation correction. Left: mean image from a rotating sample, distorted by line-scanning during motion. Right: same data after line-by-line derotation, with structures restored to their correct positions.](figure1.png)

# Statement of Need

Any imaging modality that acquires data sequentially, such as multi-photon microscopy, is susceptible to motion artifacts if the sample moves during the acquisition of a single frame. When this motion is rotational, it produces characteristic "fan-like" distortions that corrupt the morphological features of the imaged structures. This significantly complicates, or even prevents, critical downstream processing steps like cell segmentation and automated region-of-interest tracking.

This problem is particularly acute in systems neuroscience, where researchers increasingly combine two-photon or three-photon calcium imaging with behavioral paradigms involving head rotation to study sensory integration and navigation [1-4]. In such experiments, where head-fixed animals may be passively rotated or actively turn, high-speed angular motion can render imaging data unusable without correction. While individual labs have implemented custom scripts to address this, there has been no validated, open-source, and easy-to-use Python tool available to the broader community.

derotation directly fills this gap by providing a documented, tested, and modular solution for post hoc correction of imaging data acquired during rotation. It empowers researchers to perform quantitative imaging during high-speed rotational movements without requiring modifications to their existing acquisition hardware. The package is especially valuable for imaging modalities with lower frame rates, such as three-photon microscopy, where rotational distortions within a single frame are more pronounced. By providing a robust and accessible tool, derotation lowers the barrier for entry into complex behavioral experiments and improves the reproducibility of a key analysis step in a growing field of research.

# Functionality
The core of the derotation package is a line-by-line affine transformation. It operates by first establishing a precise mapping between each scanned line in the movie and the rotation angle of the sample at that exact moment in time. It then applies an inverse rotation transform to each line around a specified or estimated center of rotation. Finally, the corrected lines are reassembled into frames, producing a movie that appears as if the sample had remained stationary.

## Data Ingestion and Synchronization
The package is designed to be flexible with respect to input data formats. The primary input is the raw imaging movie (as a TIFF stack) and timing information. To determine the rotation angle for each line, the software can directly parse raw hardware signals. Through the pipelines described below, processes analog or digital recordings of the microscope's line_clock and frame_clock along with signals from the rotation motor (e.g., rotation_clock ticks). For users who have already pre-processed this information, the package can instead accept a simple NumPy array containing the cumulative rotation angle for each line of the movie.

## Processing Pipelines
For ease of use, derotation provides two high-level processing workflows tailored to common experimental paradigms. These pipelines handle data loading, parameter validation, processing, and saving outputs, while also generating logs and debugging plots to ensure quality control.

- The FullPipeline is engineered for complex experimental paradigms involving randomized, bi-directional (back-and-forth) rotations. As part of its workflow, it can optionally estimate the center of rotation automatically using Bayesian optimization, which minimizes residual motion in the corrected movie.

-The IncrementalPipeline is optimized for continuous, single-direction rotations. It assumes the center of rotation is known and provided by the user, which allows for faster processing as it bypasses the optimization step.

Both pipelines are configurable via YAML files or Python dictionaries, promoting reproducible analysis by making it straightforward to document and re-apply the same parameters across multiple datasets.

## Modularity and Outputs
While the pipelines offer a convenient workflow, the package is modular by design. The core transformation logic is accessible through the derotate_an_image_array_line_by_line function, allowing advanced users to integrate the derotation algorithm into their own custom analysis scripts.

Upon completion, a pipeline run generates a comprehensive set of outputs:

- The primary corrected movie, saved as a TIFF stack.
- A log file detailing the calculated rotation angles for each line.
- Diagnostic plots, such as angle-versus-time, to help validate the signal processing.
- Quality-control metrics and plots to assess the success of the correction.

## Validation and Extensibility
The package's effectiveness has been validated on both synthetic datasets, where the ground-truth geometry is known, and on real three-photon recordings from head-rotated mice. In both cases, the corrected images showed restored cellular morphology and were successfully processed by standard downstream analysis pipelines such as Suite2p [@Pachitariu2017] without residual rotational artifacts.

For further testing and development, the package includes a Rotator class that can generate synthetic distorted movies from a static source image, simulating various experimental conditions (e.g., different frame rates, rotation speeds, or scan patterns). This allows users to explore the parameter space of the problem and validate the correction method on their own terms.

The implementation relies on standard Python scientific libraries, including NumPy, SciPy, and Scikit-optimize, and is distributed under a BSD-3-Clause license. Comprehensive documentation, tutorials, and example datasets are available at https://derotation.neuroinformatics.dev. Using Binder, users can run the software in a cloud-based environment with sample data without requiring any local installation.

# Acknowledgements

We thank Simon Weiler for providing three-photon imaging datasets used during development and testing, and the Neuroinformatics Unit at the Sainsbury Wellcome Centre for infrastructure and support.

## References
Previous work on derotation of calcium imaging movies:
- 1. [Velez-Fort et al., 2018, Neuron](https://doi.org/10.1016/j.neuron.2018.02.023)
- 2. [Hennestad et al., 2021, Cell Reports](https://doi.org/10.1016/j.celrep.2021.110134)
- 3. [Sit & Goard, 2023, Nature Communications](https://doi.org/10.1038/s41467-023-37704-5)
- 4. [Voigts & Harnett, 2020, Neuron](https://doi.org/10.1016/j.neuron.2019.10.016)
- 5. [Pachitariu et al., 2016, BioRxiv](https://doi.org/10.1016/j.neuron.2017.07.007)

This package was inspired by [previous MATLAB script on derotation](https://github.com/jvoigts/rotating-2p-image-correction).
