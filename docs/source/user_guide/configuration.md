(user_guide/configuration)=
# Configuration Guide

This page explains how to configure Derotation for use with the full processing pipeline, including the handling of analog signals and rotation metadata.

## When is configuration required?

You need to provide a configuration file if you are using the **full pipeline** or the **incremental pipeline**. This configuration can be passed as a Python `dict`

If you only want to use the **core derotation method** — {func}`derotation.derotate_by_line.derotate_an_image_array_line_by_line` — then you must manually prepare the image stack and the angle array. In this case, configuration is not needed.


## Config Structure

Here’s a breakdown of the main configuration sections and what they control:

### `paths_read`
Paths to the input files:
- `path_to_randperm`: stimulus randomization `.mat` or `.csv` file
- `path_to_aux`: `.npy` or `.bin` file with analog signals
- `path_to_tif`: raw image data

### `paths_write`
Output locations:
- `debug_plots_folder`: for diagnostic plots
- `logs_folder`: for logs and processing info
- `derotated_tiff_folder`: where the output stack is saved
- `saving_name`: base name for output TIFF

### `channel_names`
List of signal names expected in the analog file. Matters only if using `.bin` format for analog signals.

### `rotation_increment`
Expected angle increment per tick from the motor (in degrees).

### `adjust_increment`
If `True`, the pipeline may slightly adjust the rotation increment value to match observed data. This is necessary as the motor may not exactly provide the expected number of ticks for each rotation.

### `rot_deg`
Total degrees corresponding to a full rotation (typically 360).

### `debugging_plots`
Whether to save intermediate plots for inspection. Recommended for debugging.

### `frame_rate`
Frame rate of acquisition (in Hz).

### `analog_signals_processing`
Parameters used to extract information from analog signals:
- `find_rotation_ticks_peaks` (`height`, `distance`): parameters for detecting rotation ticks
- `squared_pulse_k`: threshold for detecting line and frame clock signals
- `inter_rotation_interval_min_len`: minimum number of samples between distinct rotations
- `angle_interpolation_artifact_threshold`: threshold for discarding rotation segments with high noise

### `interpolation`
Specifies how to compute rotation angles:
- `line_use_start`: if True, use line start time for angle interpolation
- `frame_use_start`: same logic for frame-based interpolation

### Center of rotation optimization (optional)
Only needed if using **Bayesian Optimization** to refine the center:
- `biased_center`: initial guess for the rotation center `[x, y]`
- `delta_center`: max deviation in pixels from the biased center
- `init_points`: number of random points for the optimizer to try first
- `n_iter`: number of BO iterations
