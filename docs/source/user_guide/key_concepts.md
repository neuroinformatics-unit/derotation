(user_guide/key_concepts)=
# Key Concepts

`derotation` is a modular package for reconstructing rotating image stacks acquired via line scanning microscopes. This page covers the core ideas, flow of information, and how the main modules interact.
See the [API documentation](api_index) for full reference.

## How derotation works

Without derotation, movies acquired with a line scanning microscope are geometrically distorted — each line is captured at a slightly different angle, making it hard to register or interpret the resulting images.

If the angle of rotation is recorded, `derotation` can reconstruct each frame by assigning a rotation angle to each acquired line and rotating it back to its original position. This is what is called derotation-by-line. This process incrementally reconstructs each frame and can optionally include shear deformation correction.

![Derotation process](../_static/derotation_by_line.gif)

## How to use derotation

There are two main ways to use `derotation`:

### 1. **Low-level core function**
Use `derotate_an_image_array_line_by_line` to derotate an image stack, given:
- The original calcium imaging movie (expects only one imaging plane)
- Rotation angle per line

This is ideal for testing and debugging with synthetic or preprocessed data.

### 2. **Full or incremental pipeline classes**
Use the pre-made pipelines to run end-to-end processing:

- `FullPipeline`: assumes randomised complete clockwise and counter-clockwise rotations. It includes:
  - Analog signal parsing
  - Angle interpolation
  - Bayesian optimization for center estimation
  - Derotation by line (calling `derotate_an_image_array_line_by_line`)

- `IncrementalPipeline`: assumes a continuous rotation performed in small increments. It inherits functionality from the `FullPipeline` but does not perform Bayesian optimization. It can be useful as an alternative way to estimate center of rotation.

Both pipelines accept a configuration dictionary (see the [configuration guide](configuration)) and output:
- Derotated TIFF
- CSV with rotation angles and metadata
- Debugging plots
- A text file with optimal center of rotation
- Logs

You can subclass `FullPipeline` to create custom pipelines by overwriting relevant methods.

### Example usage of FullPipeline:

```python
from derotation.config.load_config import update_config_paths, load_config
from derotation.analysis.full_derotation_pipeline import FullPipeline


# Load the configuration file
pipeline_type = "full"
config = load_config(pipeline_type)
config = update_config_paths(
    config=config,
    tif_path="/my/data/image_stack.tif",
    bin_path="/my/data/signals.bin",
    dataset_path="/my/data/",
    kind=pipeline_type,
)

# Load data and run the pipeline
pipeline = FullPipeline(config)
pipeline()
```

---

## Finding the center of rotation

An accurate center of rotation is crucial for high-quality derotation. Even small errors can produce residual motion in the derotated movie — often visible as circles traced by stationary objects as cells.

![Residual motion](../_static/wrong_center.png)

In this picture, you can see in red the centers of one of the cells in the movie across different angles. Since the center is not correctly estimated, the cell appears to move in a circle.

Derotation offers two approaches for estimating the center:

### Bayesian optimization via FullPipeline

This method searches for the correct center of rotation by derotating the whole movie and minimizing a custom metric, computed through the function `ptd_of_most_detected_blob`. It requires the average image of the derotated movie at different rotations angles, and from them detects blobs, searches for the most frequent and calculates the Point-to-Point Distance (PTD) for it across blob centers at different rotation angles.

It is robust but computationally expensive.

### Ellipse fitting via IncrementalPipeline

This method exploits the fact that incremental datasets rotate very slowly and smoothly. It works by:

- Detecting largest blob in the first frame
- Tracking its position across rotations
- Fitting an ellipse to the trajectory

The center of the ellipse is assumed to match the true center of rotation. This method fails when the cell stops being visible in certain rotation angles.

Once the center is estimated, it can be fed to the FullPipeline to derotate the whole movie.

#### Example usage of IncrementalPipeline:

```python
from derotation.config.load_config import update_config_paths, load_config
from derotation.analysis.incremental_derotation_pipeline import IncrementalPipeline
from derotation.analysis.full_derotation_pipeline import FullPipeline


# Load the configuration file
pipeline_type = "incremental"
config_incremental = load_config(pipeline_type)
config_incremental = update_config_paths(
    config=config,
    tif_path="/my/data/image_stack.tif",
    bin_path="/my/data/signals.bin",
    dataset_path="/my/data/",
    kind=pipeline_type,
)

# Load data and run the pipeline
incremental_pipeline = IncrementalPipeline(config_incremental)
incremental_pipeline()

# Load the configuration file
pipeline_type = "full"
config_full = load_config(pipeline_type)
config_full = update_config_paths(
    config=config,
    tif_path="/my/data/image_stack.tif",
    bin_path="/my/data/signals.bin",
    dataset_path="/my/data/",
    kind=pipeline_type,
)

# Load data and run the pipeline
full_pipeline = FullPipeline(config_full)
full_pipeline.center_of_rotation = incremental_pipeline.center_of_rotation
full_pipeline()
```

---

## Verifying derotation quality

Use debugging plots and logs to assess the quality of your reconstruction. These include:
- Analog signal overlays
- Angle interpolation traces
- Center estimation visualizations
- Derotated frame samples
Debugging plots are by default saved in the ``debug_plots`` folder.

### Custom plotting hooks
To monitor what is happening at every step of line-by-line derotation, you can use custom plotting hooks. These are functions that are called at specific points in the pipeline and can be used to visualize intermediate results.

There are two steps in the ``derotate_an_image_array_line_by_line`` function where hooks can be injected:
- After adding a new line to the derotated image (`plotting_hook_line_addition`)
- After completing a frame (`plotting_hook_image_completed`)

Here and example of two pre-made hooks that can be used to visualize the derotation process:
```python
from derotation.plotting_hooks.for_derotation import image_completed, line_addition

hooks = {
    "plotting_hook_line_addition": line_addition,
    "plotting_hook_image_completed": image_completed,
}

pipeline = FullPipeline(config)
pipeline.hooks = hooks
pipeline()
```

> ⚠️ Note: Hooks may slow down processing significantly. Use them for inspection only.
You can also inject **custom plotting hooks** at defined pipeline stages. See the examples page for a demonstration. *Note: hooks may significantly slow down processing.*

---

## Simulated data

Use the `Rotator` and `SyntheticData` classes to generate test data:

- `Rotator`: applies line-by-line rotation to an image stack, simulating a rotating microscope. It can be used to generate challenging synthetic data that include wrong centers of rotation and out of imaging plane rotations.
- `SyntheticData`: creates fake cell images, assigns rotation angles, and generates synthetic stacks. This is especially useful for validating both the incremental and full pipelines under known conditions.

![Rotator](../_static/rotator.gif)

This is an example of a synthetic dataset with two cells generated with the `Rotator` class.

---


## Limitations

Derotation supports two experimental configurations: randomized full rotations (in the `FullRotation` pipeline) and small-step incremental rotations (`IncrementalPipeline`). Other rotation paradigms are not currently supported out of the box.

The package assumes strict input formats — TIFF stacks for images and `.bin` files with analog signals following a specific channel order. Both pipelines require:
- timing of rotation ticks, which are used to compute rotation angles;
- line clock signals, which indicate the start of a new line;s
- frame clock signals, which indicate the start of a new frame;
- a rotation on signal, which indicates when the rotation is happening.

If your data is stored in different formats or structured differently, you can write a **custom data loader** that loads rotation angles and line/frame timing, then passes them directly to the core derotation function or integrates into a custom pipeline subclass.

If you don't have a step motor but a continuous array of rotation angles, you have to clean the signal and interpolate it to match the line clock signal. You would have to write a custom data loader to handle this and a pipeline subclass to process the data.
