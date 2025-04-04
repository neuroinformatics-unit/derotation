"""
Visualizing derotation line-by-line with custom hooks
=====================================================

This example demonstrates how to use a custom plotting hook to visualize how
each image frame is reconstructed during derotation, line by line.

The derotation process works by rotating each horizontal line of an image by a
given angle and placing it into a new derotated frame. Understanding this
process is useful both for debugging and for developing intuition about how
motion correction works.

In this example, we define a custom hook called after each line is processed
during derotation. The hook visualizes the following:
    - The original image, with the current line highlighted
    - The partially built derotated image
    - The currently rotated line, overlaid on the right-hand image (if
      visible)

The hook is triggered only for frame 135 and every 20 lines, to keep the output
readable.

This can be particularly helpful when:
    - You want to verify that the rotation mapping is working as expected
    - You want to debug center of rotation or interpolation artifacts
    - You are developing your own hooks or modifying the pipeline internals
"""

# %%
# Imports
# -------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.config.load_config import load_config, update_config_paths

# %%
# Load and configure paths
# ------------------------
# We'll use a small example dataset and write output to the current folder.

current_module_path = Path.cwd()
data_folder = current_module_path / "data"
output_folder = current_module_path

config = load_config()
config = update_config_paths(
    config=config,
    tif_path=str(data_folder / "rotation_sample.tif"),
    aux_path=str(data_folder / "analog_signals.npy"),
    stim_randperm_path=str(data_folder / "stimulus_randperm.csv"),
    output_folder=str(output_folder),
)

# %%
# Define a custom hook function
# -----------------------------
# This hook is called after every line is derotated and placed into the new
# frame. We'll use it to inspect how frame 135 is constructed over time.
# This is a simplified version of the hook
# :func:`derotation.plotting_hooks.for_derotation.line_addition`.


def inspect_frame_135_and_line_180(
    derotated_filled_image: np.ndarray,
    rotated_line: np.ndarray,
    image_counter: int,
    line_counter: int,
    angle: float,
    original_image: np.ndarray,
):
    if image_counter == 135 and line_counter == 180:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        #  background fig color: black
        fig.patch.set_facecolor("black")

        ax[0].imshow(original_image, cmap="turbo")
        #  highlight the line in the original image
        ax[0].plot(
            [0, original_image.shape[1] - 1],
            [line_counter, line_counter],
            color="red",
            linewidth=2,
        )
        ax[0].set_title(
            f"Take line {line_counter}\nfrom original image,\n then rotate it"
            f"of {angle:.2f} degrees"
        )
        ax[0].title.set_color("white")
        ax[0].axis("off")

        ax[1].imshow(derotated_filled_image, cmap="turbo")
        ax[1].set_title(
            f"Place the line in a new image\nto build frame {image_counter}"
        )
        ax[1].title.set_color("white")
        ax[1].axis("off")

        #  plot on top axis 1 the rotated_line with a red colormap
        ax[1].imshow(rotated_line, cmap="hsv", alpha=0.5)

        plt.show()


# %%
# Register the hook
# -----------------

hooks = {
    "plotting_hook_line_addition": inspect_frame_135_and_line_180,
}

# %%
# Run the derotation pipeline with the custom hook
# ------------------------------------------------
# Our hook will be called during processing of each line.

pipeline = FullPipeline(config)
pipeline.hooks = hooks
pipeline()

# %%
