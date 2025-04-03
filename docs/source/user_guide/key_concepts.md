(user_guide/key_concepts)=
# Key Concepts

`derotation` is a modular package for reconstructing rotating image stacks acquired via line scanning microscopes. This page covers the core ideas, flow of information, and how the main modules interact.
See the [API documentation](../api_index.rst) for a detailed overview of the package.

## How derotation works

Without derotation, movies acquired with a line scanning microscope are geometrically distorted — each line is captured at a slightly different angle, making it hard to register or interpret the resulting images.

```{figure} ../_static/derotation_index.png

In this figure you can see on the left a schematic of a line-scanning microscope acquiring an image of a grid. The scanning pattern plus the sample rotation lead fan-like aftefacts. On the right you see a grid that has been imaged while still (top), while rotating (middle) and the derotated image (bottom). The grid is now perfectly aligned.
```


If the angle of rotation is recorded, `derotation` can reconstruct each frame by assigning a rotation angle to each acquired line and rotating it back to its original position. This is what is called derotation-by-line. This process incrementally reconstructs each frame and can optionally include shear deformation correction.


```{figure} ../_static/derotation_by_line.gif

In this animation you can see the reconstruction of a frame with derotation. The original frame with deformation is on the left. With subsequent iterations, derotation picks a line, rotates it according to the calculated angle, and adds it to the new derotated frame. The final result is on the right.
```


## How to use derotation

There are two main ways to use `derotation`:

### 1. **Low-level core function**
Use {func}`derotation.derotate_by_line.derotate_an_image_array_line_by_line` to derotate an image stack, given:
- The original multi-photon movie (expects only one imaging plane)
- Rotation angle per line

This is ideal for testing and debugging with synthetic or preprocessed data.

Please refer to the [example on how to use the core function](../examples/core_fucntion.rst) for a demonstration of how to use {func}`derotation.derotate_by_line.derotate_an_image_array_line_by_line`.

### 2. **Full and Incremental Pipeline Classes**

Derotation provides two pre-built pipelines for end-to-end processing:

- **{class}`derotation.analysis.full_derotation_pipeline.FullPipeline`**
  Assumes randomized, alternating clockwise and counter-clockwise rotations. It performs:
  - Analog signal parsing
  - Angle interpolation
  - Bayesian optimization for center estimation
  - Line-by-line derotation using {func}`derotation.derotate_by_line.derotate_an_image_array_line_by_line`

- **{class}`derotation.analysis.incremental_derotation_pipeline.IncrementalPipeline`**
  Assumes a continuous rotation performed in small increments. Inherits from `FullPipeline` but skips Bayesian optimization.
  Useful when the center of rotation is known or estimated differently.

Both pipelines accept a **configuration dictionary** (see the [configuration guide](./configuration.md)) and produce:
- A derotated TIFF stack
- A CSV file with rotation angles and metadata
- Debugging plots
- A text file containing the estimated optimal center of rotation
- Log files

You can create custom pipelines by subclassing `FullPipeline` and overriding the relevant methods.

See the [usage example](../examples/pipeline_with_real_data.zip) for how to instantiate `FullPipeline`, run it, and inspect its attributes and outputs.

---

### Data Format and Requirements

These pipelines are designed for a specific experimental setup. They expect analog input signals in a fixed order and may not work out-of-the-box with custom data.

The following inputs are required:

- A **numpy array** of analog signals, with four channels in this order:
  1. **Line clock** – signals the start of a new line (from ScanImage)
  2. **Frame clock** – signals the start of a new frame (from ScanImage)
  3. **Rotation ON signal** – indicates when the motor is rotating
  4. **Rotation ticks** – used to compute rotation angles (from the step motor)

- A **CSV file** describing speeds and directions of rotation in the following format:

  ```csv
  speed,direction
  200,-1
  200,1
  ```

Refer to the [configuration guide](./configuration.md) for more details on specifying file paths and parameters.

---

## Finding the center of rotation

An accurate center of rotation is crucial for high-quality derotation. Even small errors can produce residual motion in the derotated movie — often visible as circles traced by stationary objects as cells.

```{figure} ../_static/wrong_center.png

In this picture, you can see as red crosses the centers of one of the cells in the movie across different angles. Since the center is not correctly estimated, the cell appears to move in a circle.
```

Derotation offers two approaches for estimating the center:

**Bayesian optimization via FullPipeline**

This method searches for the correct center of rotation by derotating the whole movie and minimizing a custom metric, computed through the function {func}`derotation.analysis.metrics.ptd_of_most_detected_blob`. It requires the average image of the derotated movie at different rotations angles, and from them detects blobs, searches for the most frequent and calculates the Point-to-Point Distance (PTD) for it across blob centers at different rotation angles.

It is robust but computationally expensive.

**Ellipse fitting via {class}`derotation.analysis.incremental_derotation_pipeline.IncrementalPipeline`**

This method exploits the fact that incremental datasets rotate very slowly and smoothly. It works by:

- Detecting largest blob in the first frame
- Tracking its position across rotations
- Fitting an ellipse to the trajectory

The center of the ellipse is assumed to match the true center of rotation. This method fails when the cell stops being visible in certain rotation angles.

Once the center is estimated, it can be fed to the {class}`derotation.analysis.full_derotation_pipeline.FullPipeline` to derotate the whole movie.

---

## Verifying derotation quality

Use debugging plots and logs to assess the quality of your reconstruction. These include:
- Analog signal overlays
- Angle interpolation traces
- Center estimation visualizations
- Derotated frame samples
Debugging plots are by default saved in the ``debug_plots`` folder.

You can see some of the debugging plots in the [example using real data](../examples/pipeline_with_real_data.rst) and the [example to find the center of rotation with synthetic data](../examples/find_center_of_rotation.rst).

### Custom plotting hooks
To monitor what is happening at every step of line-by-line derotation, you can use custom plotting hooks. These are functions that are called at specific points in the pipeline and can be used to visualize intermediate results.

There are two steps in the {func}`derotation.derotate_by_line.derotate_an_image_array_line_by_line` function where hooks can be injected:
- After adding a new line to the derotated image ({func}`derotation.plotting_hooks.for_derotation.line_addition`)
- After completing a frame ({func}`derotation.plotting_hooks.for_derotation.image_completed`)
-
> ⚠️ Note: Hooks may slow down processing significantly. Use them for inspection only.
You can also inject **custom plotting hooks** at defined pipeline stages. See the examples page for a demonstration. *Note: hooks may significantly slow down processing.*

See the [plotting hooks example](../examples/use_plotting_hooks.rst) for a demonstration of how to use a custom plotting hook.

---

## Simulated data

Use the {class}`derotation.simulate.line_scanning_microscope.Rotator` and {class}`derotation.simulate.synthetic_data.SyntheticData` classes to generate test data:

- {class}`derotation.simulate.line_scanning_microscope.Rotator`: applies line-by-line rotation to an image stack, simulating a rotating microscope. It can be used to generate challenging synthetic data that include wrong centers of rotation and out of imaging plane rotations.
- {class}`derotation.simulate.synthetic_data.SyntheticData`: creates fake cell images, assigns rotation angles, and generates synthetic stacks. This is especially useful for validating both the incremental and full pipelines under known conditions.

```{figure} ../_static/rotator.gif

This animation shows the synthetic data generated with the Rotator class. As you can see these are two "cells" that rotate around a given center of rotation.
```

This is an example of a synthetic dataset with two cells generated with the {class}`derotation.simulate.line_scanning_microscope.Rotator` class.

You can find different examples on how to use the Rotator and SyntheticData classes in the [examples page](../examples/index.rst):
- [Use the Rotator to create elliptically rotated data](../examples/elliptical_rotations.rst)
- [Find center of rotation with synthetic data](../examples/find_center_of_rotation.rst)
- [Simple rotation and derotation of an image stack](../examples/rotate_and_derotate_a_square.rst)

---
