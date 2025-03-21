(user_guide/key_concepts)=
# Key Concepts

`derotation` is a modular package for reconstructing rotating image stacks acquired via line scanning microscopes. This page covers the core ideas, flow of information, and how the main modules interact.


## How derotation works

Without derotation, movies acquired with a line scanning microscope are geometrically distorted — each line is captured at a slightly different angle, making it hard to register or interpret the resulting images.

If the angle of rotation is recorded, `derotation` can reconstruct each frame by assigning a rotation angle to each acquired line and rotating it back to its original position. This process incrementally reconstructs each frame and can optionally include shear deformation correction.


## How to use derotation

There are two main ways to use `derotation`:

### 1. **Low-level core function**
Use `derotate_an_image_array_line_by_line` to derotate an image stack, given:
- Rotation angle per line
- Center of rotation

This is ideal for testing and debugging with synthetic or preprocessed data.

### 2. **Full or incremental pipeline classes**
Use the pre-made pipelines to run end-to-end processing:

- `FullPipeline`: assumes randomised complete clockwise and counter-clockwise rotations. It includes:
  - Analog signal parsing
  - Angle interpolation
  - Bayesian optimization for center estimation

- `IncrementalPipeline`: assumes a continuous rotation performed in small increments. It inherits fuctionality from the `FullPipeline` but does not perform bayesian optimization. It can be useful to rack brightness across angles and estimate center via blob detection and ellipse fitting.

Both pipelines accept a configuration dictionary (see the [configuration guide](configuration)) and output:
- Derotated TIFF
- CSV with rotation angles and metadata
- Debugging plots
- Logs

You can subclass `FullPipeline` to create custom pipelines by overwriting relevant methods.

---

## Finding the center of rotation

A small error in the rotation center can result in visible residual motion (e.g. cells drawing arcs).

Two approaches are provided:

- **Ellipse fitting via IncrementalPipeline**: tracks bright blobs and fits ellipses to their trajectories.
- **Bayesian optimization via FullPipeline**: minimizes the Point-to-Point Distance (PTD), a metric quantifying instability in pixel intensities across time.

---

## Verifying derotation quality

Use debugging plots and logs to assess the quality of your reconstruction. These include:
- Analog signal overlays
- Angle interpolation traces
- Center estimation visualizations
- Derotated frame samples

You can also inject **custom plotting hooks** at defined pipeline stages. See the examples page for a demonstration. *Note: hooks may significantly slow down processing.*

---

## Simulated data

Use the `Rotator` and `SyntheticData` classes to generate test data:

- `Rotator`: applies line-by-line rotation to an image stack, simulating a rotating microscope.
- `SyntheticData`: creates fake cell images, assigns rotation angles, and generates synthetic stacks. This is especially useful for validating both the incremental and full pipelines under known conditions.

---

See the [API documentation](api_index) for full reference.

## Limitations

Derotation supports two experimental configurations: randomized full rotations (in the `FullRotation` pipeline) and small-step incremental rotations (`IncrementalPipeline`). Other rotation paradigms are not currently supported out of the box.

The package assumes strict input formats — TIFF stacks for images and `.bin` files with analog signals following a specific channel order. Both pipelines require:
- timing of rotation ticks, which are used to compute rotation angles;
- line clock signals, which indicate the start of a new line;s
- frame clock signals, which indicate the start of a new frame;
- a rotation on signal, which indicates when the rotation is happening.

If your data is stored in different formats or structured differently, you can write a **custom data loader** that loads rotation angles and line/frame timing, then passes them directly to the core derotation function or integrates into a custom pipeline subclass.