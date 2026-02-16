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
    orcid: 0009-0004-4816-321X
    affiliation: "1"
  - name: Simon Weiler
    orcid: 0000-0003-4731-8369
    affiliation: "1"
  - name: Adam L. Tyson
    orcid: 0000-0003-3225-1130
    affiliation: "2"
  - name: Troy W. Margrie
    orcid: 0000-0002-5526-4578
    affiliation: "1"
affiliations:
  - name: The Sainsbury Wellcome Centre for Neural Circuits and Behaviour, University College London, 25 Howland Street, London W1T 4JG, UK
    index: 1
  - name: Neuroinformatics Unit, Sainsbury Wellcome Centre & Gatsby Computational Neuroscience Unit, University College London, London W1T 4JG, UK
    index: 2
date: 13 February 2026
bibliography: paper.bib
---


# Summary

In line-scanning microscopy, sample rotation during acquisition introduces geometric distortions that compromise downstream analyses such as cell detection and signal extraction. While several groups have developed custom solutions [@velez-fort_circuit_2018; @hennestad_mapping_2021; @sit_coregistration_2023; @voigts_somatic_2020], a general-purpose tool has been lacking. `derotation` is an open-source Python package that corrects these artifacts by applying a line-by-line inverse rotation using recorded angles and the microscope's line acquisition clock, restoring the expected geometry and enabling reliable cell segmentation during rapid rotational movements (Figure 1).

![Example of `derotation` correction. On the left, the average of a series of images acquired using 3-photon microscopy of layer 6 mouse cortical neurons labeled with GCaMP7f during passive rotation. Center, the mean image after derotation, and on the left the mean image of the derotated movie after suite2p registration [@pachitariu_suite2p_2016]. As you can see, following derotation the cells are visible and have well defined shapes.](figure1.png)

# Statement of Need

When sample motion during acquisition is rotational, it produces characteristic "fan-like" distortions that corrupt the morphological features of the imaged structures (Figure 2), significantly complicating or preventing cell segmentation and automated region-of-interest tracking.

This problem is particularly acute in systems neuroscience, where researchers increasingly combine multiphoton calcium imaging with behavioral paradigms involving head rotation [@velez-fort_circuit_2018; @hennestad_mapping_2021; @sit_coregistration_2023; @voigts_somatic_2020]. High-speed angular motion can render imaging data unusable, especially for modalities with lower frame rates such as three-photon imaging. Despite multiple lab-specific solutions, no validated, open-source, and easy-to-use Python tool has been available to the broader community.

![Schematic of line-scanning microscope distortion. Left: line scanning pattern plus sample rotation lead to fan-like artifacts when imaging a grid. Right: grid imaged while still (top), while rotating at 200°/s with 7Hz frame rate (middle), and after `derotation` (bottom), showing alignment restoration.](figure2.png)

`derotation` meets this need by providing a documented, tested, and modular solution for post hoc correction of imaging data acquired during rotation. The package has been developed openly on GitHub since June 2023, with contributions from five developers. It is distributed on PyPI under a BSD-3-Clause license, with comprehensive documentation, tutorials, and runnable examples via Binder at https://derotation.neuroinformatics.dev. By providing a robust and accessible tool, `derotation` lowers the barrier for complex behavioral experiments and improves the reproducibility of a key analysis step in a growing field of research.

The package has been validated on three-photon recordings of deep cortical neurons expressing the calcium indicator GCaMP7f in head-fixed mice (Figure 3). The corrected images showed restored cellular morphology and were successfully processed by Suite2p [@pachitariu_suite2p_2016]. Compared with frame-by-frame affine correction, line-by-line derotation preserves ROI fluorescence signals during rotation periods, eliminating the artificial dips visible in Figure 3.

![Figure 3. Validation on 3-photon data. Left: mean image after line‑by‑line derotation. Red circle marks the ROI used for the plots on the right. Top right: sample $\Delta F/F_0$ timecourse for the selected ROI (pink = line‑by‑line derotation; gray = frame‑by‑frame affine correction; shaded vertical bars = rotation intervals). Bottom right: mean $\Delta F/F_0$ aligned to rotation periods for clockwise and counterclockwise rotations. Line‑by‑line derotation preserves the ROI signal during rotations and removes the artificial dips introduced by frame‑by‑frame correction. Clockwise and counterclockwise traces show a roughly mirror‑symmetric, angle‑dependent modulation of measured fluorescence with the frame-by-frame correction.](figure3.png)

# State of the Field

Several groups have developed rotation correction procedures as part of their experimental pipelines [@velez-fort_circuit_2018; @hennestad_mapping_2021; @sit_coregistration_2023], but each solution remains embedded in a lab-specific workflow, typically applying frame-level corrections in MATLAB. The closest prior work is @voigts_somatic_2020, who described a line-by-line derotation approach for two-photon imaging during free locomotion, but their code was not designed to generalize across setups with lower frame rates. General-purpose image registration tools such as Suite2p [@pachitariu_suite2p_2016] correct for translational motion but cannot handle within-frame rotational distortions. `derotation` fills this gap by integrating with the scientific Python ecosystem and producing output that can be directly fed into standard registration pipelines.

# Software Design

`derotation` separates the core algorithm from experiment-specific logic through a layered, object-oriented design. The core function (`derotate_an_image_array_line_by_line`) takes a movie and a per-line angle array and returns the corrected stack, with no dependencies on configuration or I/O, so it can be used directly in any custom workflow.

On top of this core, two pipeline classes (`FullPipeline` and `IncrementalPipeline`) orchestrate the end-to-end processing: data loading, angle interpolation, optional center-of-rotation estimation, derotation, and output saving. Each processing step is implemented as an overridable method, so users can subclass either pipeline to adapt to new experimental setups or data formats. A simulation module generates synthetic test data for systematic regression testing. Full API documentation, configuration guides, and runnable examples are available at https://derotation.neuroinformatics.dev.

# Research Impact Statement

`derotation` is in use at the Sainsbury Wellcome Centre to process three-photon calcium imaging data acquired during passive head rotation experiments, with its output fed into Suite2p [@pachitariu_suite2p_2016] for cell detection and signal extraction.

# AI Usage Disclosure

GitHub Copilot was used for code autocompletion during development and to assist with drafting portions of this manuscript and the documentation. All AI-generated content was reviewed, tested, and validated by the authors, who carried out the algorithmic design, architectural decisions, and scientific validation.

# Methodological appendix

The package has been directly tested on 3-photon imaging data obtained from cortical layer 6 callosal-projecting neurons expressing the calcium indicator GCaMP7f in the mouse visual cortex. More specifically, wild type C57/B6 mice were injected with retro AAV-hSyn-Cre (1 × $10^{14}$ units per ml) in the left and with AAV2/1.syn.FLEX.GCaMP7f (1.8 × $10^{13}$ units per ml) in the right primary visual cortex. A cranial window was implanted over the right hemisphere, and then a headplate was cemented onto the skull of the animal. After 4 weeks of viral expression and recovery, animals were head-fixed on a rotation platform driven by a direct-drive motor (U-651, Physik Instrumente). 360 degree clockwise and counter-clockwise rotations with different peak speed profiles (50, 100, 150, 200 deg/s) were performed while imaging awake neuronal activity using a 25x objective (XLPLN25XWMP2, NA 1.05, 25, Olympus). Imaging was conducted at 7 Hz with 256x256 pixels. All experimental procedures were approved by the Sainsbury Wellcome Centre (SWC) Animal Welfare and Ethical Review Body (AWERB). For detailed description of the 3-photon power source and imaging please see [@cloves_vivo_2024].

# Acknowledgements

We thank Eivind Hennestad for initial project discussion. We thank Mateo Vélez-Fort and Chryssanthi Tsitoura for their assistance in building and testing the the three-photon imaging and rotation setup as well as for feedback on the package. We also thank Igor Tatarnikov for contributing to the development of the package and the whole Neuroinformatics Unit. This package was inspired by previous work on `derotation` as described in [@voigts_somatic_2020]. The authors are grateful to the support staff of the Neurobiological Research Facility at the Sainsbury Wellcome Centre (SWC). This research was funded by the Sainsbury Wellcome Centre core grant from the Gatsby Charitable Foundation (GAT3361) and Wellcome Trust (219627/Z/19/Z), a Wellcome Trust Investigator Award (214333/Z/18/Z) and Discovery Award (306384/Z/23/Z) to T.W.M. and by SWC core funding to the Neurobiological Research Facility. S.W. was funded by a Feodor-Lynen fellowship from the Alexander von Humboldt Foundation.

# References
