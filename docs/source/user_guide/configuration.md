(user_guide/configuration)=
# Configuration Guide

This page explains how to configure Derotation for use with the full processing pipeline, including the handling of analog signals and rotation metadata.

## When is configuration required?

You need to provide a configuration file if you are using the **full pipeline**, which:
- Parses analog signals (line/frame clocks, motor ticks)
- Estimates rotation angles over time
- Derotates images using either full or incremental strategies
- (Optionally) runs Bayesian Optimization to find the center of rotation

This configuration can be passed as a Python `dict`

If you only want to use the **core derotation method** — `rotate_an_image_array_line_by_line` — then you must manually prepare the image stack and the angle array. In this case, configuration is not needed.


## Config Structure

Here’s a breakdown of the main configuration sections and what they control:

### `paths_read`
Paths to the input files:
- `path_to_randperm`: stimulus randomization `.mat` file
- `path_to_aux`: `.bin` file with analog signals
- `path_to_tif`: raw image data

### `paths_write`
Output locations:
- `debug_plots_folder`: for diagnostic plots
- `logs_folder`: for logs and processing info
- `derotated_tiff_folder`: where the output stack is saved
- `saving_name`: base name for output TIFF

### `channel_names`
List of signal names expected in the analog `.bin` file. These must match the order in which the channels were saved.

### `rotation_increment`
Expected angle increment per tick from the motor (in degrees).

### `adjust_increment`
If `True`, the pipeline may slightly adjust the rotation increment value to match observed data.

### `rot_deg`
Total degrees corresponding to a full rotation (typically 360).

### `debugging_plots`
Whether to save intermediate plots for inspection.

### `frame_rate`
Frame rate of acquisition (in Hz). Needed to disambiguate overlapping events.

### `analog_signals_processing`
Parameters used to extract information from analog signals:
- `find_rotation_ticks_peaks`: settings for detecting motor ticks
- `squared_pulse_k`: controls sharpening of pulses
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

---

## Tips
- Use relative paths where possible if working across systems.
- For quick testing, you can override only part of the config using a dict and fallback to defaults.
- Bayesian optimization can take time; use it once per dataset and reuse the result if possible.

---

For real-world examples of configuration files, see the [examples directory](../examples/index.md) or try the scripts in `examples/`.