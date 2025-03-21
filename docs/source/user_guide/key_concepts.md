(user_guide/key_concepts)=
# Key Concepts

Derotation is a modular framework for reconstructing rotating image stacks from line scanning microscopes. This page covers the core ideas, flow of information, and how the main modules interact.

---

## Why derotation?

In certain imaging experiments (e.g. 3-photon calcium imaging), the sample is rotated during acquisition. This introduces geometric distortions, as each line in the image is captured at a slightly different angle.

Derotation provides a mechanism to reverse this distortion by reconstructing each image line at its corresponding angle, ultimately recovering a coherent image.

---

## Core components

### Analog signal processing
From a `.bin` file, Derotation extracts line and frame timestamps, motor state, and rotation tick events. It then reconstructs a time-aligned angle array via interpolation.

Common problems include:
- Missing or duplicated ticks
- Timing drift
- Overlapping events (e.g. tick during frame transition)

These can be debugged with intermediate plots.

### Rotation angle reconstruction
Angles are reconstructed from stepper motor ticks. Each tick represents a fixed increment (e.g. 0.2°), and interpolation maps these ticks to precise line times. Optional smoothing or artifact filtering is applied.

### Line-by-line derotation
This is performed by:
```python
rotate_an_image_array_line_by_line(image, angle_per_line, center)
```
Each line is rotated back to its expected position using the angle and a defined center.

### Why finding the optimal center matters
A small error in the rotation center leads to large distortions, especially near image edges. Estimating the correct center is critical for proper derotation.

---

## Estimating the rotation center

### Ellipse fitting and blob detection
A bright spot is tracked through frames, and its trajectory is fit to an ellipse. The center of this ellipse estimates the rotation center. Blob detection methods locate candidate features, and filtering ensures a single clean trajectory.

### Bayesian Optimization with PTD
An alternative method uses Bayesian Optimization to minimize a metric:
- **PTD (Pixel Temporal Deviation)**: measures the temporal variance of each pixel.
- A well-aligned derotation should produce stable pixel values over time.

The optimizer searches over possible centers to minimize PTD, typically over 10–20 iterations.

---

## FullPipeline and Rotator

- `FullPipeline`: handles reading, preprocessing, signal parsing, angle reconstruction, and derotation. It accepts YAML or dict configs.
- `Rotator`: core class that performs geometric rotation, interpolation, and angle alignment.

---

## Incremental pipeline use case
The incremental pipeline is ideal when the sample is rotated in tiny steps (e.g. 0.2°), rather than in full turns. This allows building a single composite image over a sequence of steps, useful in high-resolution anatomical mapping.

---

## Logs and CSV output
Derotation saves logs and CSV summaries alongside the output TIFF. These include:
- Rotation angles per frame
- Center estimates
- Runtime diagnostics

They are saved in `logs_folder` and `derotated_tiff_folder`.

---

## Plotting hooks and debugging
Plotting hooks allow injecting custom plots at defined pipeline stages. Combined with debug flags, this helps:
- Visualize signal thresholds
- Track center estimation results
- Inspect derotated stacks

A separate Debugging Guide will expand on how to interpret and act on these plots.

---

## Synthetic data
Synthetic stacks can be created using:
```python
from derotation.simulate.synthetic_data import generate_elliptical_rotation_stack
```
This is helpful for:
- Testing alignment tools
- Validating optimization steps
- Benchmarking under known ground truth