"""
Full pipeline on real data: derotation and center of rotation estimation
========================================================================

This example shows how to run the full derotation pipeline on real data,
including: a TIFF movie, analog signals (.npy), and a CSV file with stimulus
randomization. The pipeline performs automatic estimation of the center of
rotation, derotation, and saves intermediate results and visualizations in an
output folder.

Steps performed:
    - Load configuration from a YAML file and update file paths
    - Initialize the `FullPipeline` class with:
        - A TIFF movie to derotate
        - A `.npy` file with analog signals from ScanImage and a step motor
          (ordered as: frame clock, line clock, rotation on, rotation ticks)
        - A CSV file with stimulus randomization
    - Run the full pipeline, which:
        - Interpolates rotation angles per acquired line from analog signals
        - Estimates the center of rotation using a Bayesian optimization
          approach
        - Derotates the movie based on estimated parameters
        - Saves the derotated movie and rotation angles

At the end, we visualize key plots generated during processing and explore the
results saved in the output folder.

For more details, see the
`User guide <../user_guide/key_concepts.html>`_.
"""

# %%
# Imports
# -------
from pathlib import Path

import matplotlib.pyplot as plt

from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.config.load_config import load_config, update_config_paths

# %%
# Load and update configuration
# -----------------------------
# We define paths relative to the current working directory
current_module_path = Path.cwd()
data_folder = current_module_path / "data"

config = load_config()
config = update_config_paths(
    config=config,
    tif_path=str(data_folder / "rotation_sample.tif"),
    aux_path=str(data_folder / "analog_signals.npy"),
    stim_randperm_path=str(data_folder / "stimulus_randperm.csv"),
    output_folder=str(current_module_path),
)

# %%
# Initialize pipeline
# -------------------
pipeline = FullPipeline(config)

# %%
# Peek into the loaded data
# -------------------------
print(f"Loaded movie shape: {pipeline.image_stack.shape}")
plt.imshow(pipeline.image_stack[0], cmap="viridis")
plt.title("First frame of the movie")
plt.axis("off")
plt.show()

# %%
# Useful attributes before running the pipeline
print(f"Number of frames: {pipeline.num_frames}")
print(f"Lines per frame: {pipeline.num_lines_per_frame}")
print(f"Rotation speeds: {pipeline.speed} deg/s")
print(f"Rotation direction: {pipeline.direction} (−1 = CCW, 1 = CW)")
print(f"Estimated number of full rotations: {pipeline.number_of_rotations}")

# %%
# Run the full pipeline
# ---------------------
pipeline()

# %%
# Inspecting the output
# ---------------------

# %%
# Convenience handles for later use
debug_folder = pipeline.debug_plots_folder
debug_images = sorted(debug_folder.glob("*.png"))
mean_images_folder = debug_folder / "mean_images"


def get_image_path(name):
    for img_path in debug_images:
        if name == img_path.name.split(".")[0]:
            return img_path


def show_image(path):
    img = plt.imread(path)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# %%
# Rotation detection based on analog signals
show_image(get_image_path("rotation_ticks_and_rotation_on"))

print(f"Expected number of ticks: {pipeline.number_of_rotations * 360 / 0.2}")
print(f"Detected ticks: {len(pipeline.rotation_ticks_peaks)}")
print(
    f"Adjusted rotation increment: {pipeline.rotation_increment:.3f} degrees"
)

# %%
# Interpolated rotation angles per line
# Green = frame-level angles, Yellow = interpolated per-line angles
show_image(get_image_path("rotation_angles"))

# %%
# Calculated baseline (offset) of the image in arbitrary units
print(f"Estimated image offset: {pipeline.offset}")


# %%
# Original max projection with estimated center
show_image(get_image_path("max_projection_with_center"))

# %%
# Position of the most detected cell after finding the optimal center of
# rotation. As you can see it is pretty stable.
show_image(get_image_path("most_detected_blob_centers"))

# %%
# Derotated max projection with center overlaid
# Now the cells are aligned, although registration might still be needed
show_image(get_image_path("derotated_max_projection_with_center"))

# %%
# Rotation angles and derotation metadata are accessible as a pandas DataFrame
print(pipeline.derotation_output_table.iloc[125:153])
