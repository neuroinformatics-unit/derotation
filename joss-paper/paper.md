---
title: 'Derotation: a Python package for correcting rotation-induced distortions in line-scanning microscopy'
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
    orcid: TODO
    affiliation: "1"
  - name: Adam L. Tyson
    orcid: 0000-0003-3225-1130
    affiliation: "1, 2"
  - name: Troy W. Margrie
    orcid: TODO
    affiliation: "1"
affiliations:
  - name: Sainsbury Wellcome Centre, University College London, UK
    index: 1
  - name: Gatsby Computational Neuroscience Unit, University College London, London, United Kingdom
    index: 2
date: TODO
bibliography: paper.bib
---

# Summary

Line-scanning microscopy, including multiphoton calcium imaging, is a powerful technique for observing dynamic processes at cellular resolution. However, when the imaged sample rotates during acquisition, the sequential line-by-line scanning process introduces geometric distortions. These artifacts, which manifest as shearing or curving of features, can severely compromise downstream analyses such as motion registration, cell detection, and signal extraction. While several studies have developed custom solutions for this issue [@velez-fort_circuit_2018] [@hennestad_mapping_2021] [@sit_coregistration_2023] [@voigts_somatic_2020], a general-purpose, accessible software package has been lacking.

`derotation` is an open-source Python package that algorithmically reconstructs movies from data acquired during sample rotation. By leveraging recorded rotation angles and the microscope's line acquisition clock, the software applies a precise, line-by-line inverse transformation to restore the original geometry of the imaged plane. This correction enables reliable cell segmentation during rapid rotational movements, making it possible to study yaw motion without sacrificing image quality (Figure 1).

![Example of `derotation` correction. On the left, the mean image of a 3-photon movie in which the sample was rotating. In the center, the mean image after derotation, and on the left the mean image of the derotated movie after suite2p registration [@pachitariu_suite2p_2016]. As you can see, already after derotation the cells are visible and have well defined shapes.](figure1.png)

# Statement of Need

Any imaging modality that acquires data sequentially, such as line-scanning microscopy, is susceptible to motion artifacts if the sample moves during the acquisition of a single frame. When this motion is rotational, it produces characteristic "fan-like" distortions that corrupt the morphological features of the imaged structures (Figure 2). This significantly complicates, or even prevents, critical downstream processing steps like cell segmentation and automated region-of-interest tracking.

This problem is particularly acute in systems neuroscience, where researchers increasingly combine two-photon or three-photon calcium imaging with behavioral paradigms involving head rotation [@velez-fort_circuit_2018] [@hennestad_mapping_2021] [@sit_coregistration_2023] [@voigts_somatic_2020]. In such experiments, where head-fixed animals may be passively or actively rotated, high-speed angular motion can render imaging data unusable without correction. The issue is even more acute in imaging modalities with lower frame rates, such as three-photon calcium imaging. While individual labs have implemented custom scripts to address this, there has been no validated, open-source, and easy-to-use Python tool available to the broader community.

![Schematic of line-scanning microscope distortion. Left: scanning pattern plus sample rotation lead to fan-like artifacts. Right: grid imaged while still (top), while rotating (middle), and after `derotation` (bottom), showing alignment restoration.](figure2.png)

`derotation` directly fills this gap by providing a documented, tested, and modular solution for post hoc correction of imaging data acquired during rotation. It enables researchers to perform quantitative imaging during high-speed rotational movements. By providing a robust and accessible tool, `derotation` lowers the barrier for entry into complex behavioral experiments and improves the reproducibility of a key analysis step in a growing field of research.

# Functionality
The core of the `derotation` package is a line-by-line affine transformation. It operates by first establishing a precise mapping between each scanned line in the movie and the rotation angle of the sample at that exact moment in time. It then applies an inverse rotation transform to each line around a specified or estimated center of rotation. Finally, the corrected lines are reassembled into frames, producing a movie that appears as if the sample had remained stationary.


## Data Ingestion and Synchronization
The package accepts two types of input formats depending on the processing approach:  

**For pipeline workflows (`FullPipeline` and `IncrementalPipeline`):**  
These pipelines are designed for experimental setups with synchronized rotation and imaging data. The required inputs are: 

- An array of analog signals containing timing and rotation information, typically including the start of a new line and frame, when the rotation system is active, and the rotation position feedback;    
- A **CSV file** describing speeds and directions.  

**For low-level core function:**  
Advanced users can bypass the pipeline workflows and use the core transformation function directly by providing the original movie and an array of rotation angles for each line.

This modular design allows users with custom experimental setups to integrate the `derotation` algorithm into their own analysis scripts while still benefiting from the core transformation logic.

## Processing Pipelines
For ease of use, `derotation` provides two high-level processing workflows tailored to common experimental paradigms. These pipelines handle data loading, parameter validation, processing, and saving outputs, while also generating logs and debugging plots to ensure quality control.

- `FullPipeline` is engineered for experimental paradigms involving randomized, clockwise or counter-clockwise rotations. It assumes that there will be complete 360° rotations of the sample. As part of its workflow, it can optionally estimate the center of rotation automatically using Bayesian optimization, which minimizes residual motion in the corrected movie.  

- `IncrementalPipeline` is optimized for stepwise, single-direction rotations. This rotation paradigm is useful for calibration of the luminance across rotation angles. It can also provide an alternative estimate of the center of rotation, fitting the trajectory of a cell across rotation angles.  

Both pipelines are configurable via YAML files or Python dictionaries, promoting reproducible analysis by making it straightforward to document and re-apply the same parameters across multiple datasets.

Upon completion, a pipeline run generates a comprehensive set of outputs: the corrected movie, a CSV file with rotation angles and metadata for each frame, debugging plots, a text file containing the estimated optimal center of rotation, and log files with detailed processing information.

## Synthetic Data Generation
For further development and testing, `derotation` includes a synthetic data generator that can create challenging synthetic datasets with misaligned centers of rotation and out-of-plane rotations. This feature is particularly useful for validating the robustness of the `derotation` algorithm and for developing new features.

The synthetic data can be generated using the following classes:

- `Rotator` class: Core class that applies line-by-line rotation to an image stack, simulating a rotating microscope.  
- `SyntheticData` class: Creates fake cell images, assigns rotation angles, and generates synthetic stacks leveraging the `Rotator` class. It is a complete synthetic dataset generator.

## Documentation and Installation
`derotation` is available on PyPI and can be installed with `pip install derotation`. It is distributed under a BSD-3-Clause license. Comprehensive documentation, tutorials, and example datasets are available at https://derotation.neuroinformatics.dev. Using Binder, users can run the software in a cloud-based environment with sample data without requiring any local installation.

# Future Directions
Derotation is currently used to process 3-photon movies acquired during head rotation. Future directions can include further automated pipelines for specific motorised stages and experimental paradigms.

# Acknowledgements

We thank Simon Weiler, Mateo Vélez-Fort, and Chryssanthi Tsitoura for providing three-photon imaging datasets used during development and testing, and for feedback on the package. We also thank Igor Tatarnikov for contributing to the development of the package and the whole Neuroinformatics Unit. This package was inspired by previous work on `derotation` as described in [@voigts_somatic_2020].
(TODO: Add grant information.)

# References
