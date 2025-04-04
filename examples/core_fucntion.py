"""
Derotate a TIFF movie line-by-line using precomputed rotation angles
=====================================================================

This script demonstrates how to load a 3D TIFF image stack (frames × height ×
width) and apply line-by-line derotation using a set of precomputed rotation
angles.

Each horizontal line in the image is rotated by a specific angle (in degrees),
allowing correction for sample motion or intentional rotation during
acquisition.

Inputs:
- angles_per_line.npy: A 1D NumPy array of angles in degrees, with one value
  per image line.
- rotation_sample.tif: A TIFF stack to be derotated.

The script computes the derotated movie and displays a maximum intensity
projection for quick visual inspection.

"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from derotation.derotate_by_line import derotate_an_image_array_line_by_line

# %%
# Set up
# -----------

current_module_path = Path.cwd()
data_folder = current_module_path / "data"

# %%
# Load per-line rotation angles

angles_path = data_folder / "angles_per_line.npy"
print(f"Loading per-line rotation angles from {angles_path}")
angles_per_line = np.load(angles_path)  # Shape: (height,)

# %%
# Load the TIFF stack (shape: num_frames × height × width)

tif_path = data_folder / "rotation_sample.tif"
print(f"Loading image stack from {tif_path}")
image_stack = tifffile.imread(tif_path)

print(f"Image stack shape: {image_stack.shape}")
print(f"Angles per line shape: {angles_per_line.shape}")

# %%
# See the angles
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(np.arange(angles_per_line.shape[0]), angles_per_line, s=1)
ax.set_title("Rotation angles per line")
ax.set_xlabel("Line number")
ax.set_ylabel("Angle (degrees)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.show()


# %%
# Set derotation parameters
# see :meth:`derotation.analysis.full_derotation_pipeline.FullPipeline
# .find_image_offset` for more details on finding the offset value
rotation_center = (129, 128)  # (y, x) format
blank_pixels_value = -3623  # Value for pixels outside the rotated area

print(f"Applying derotation with center: {rotation_center}")

# %%
#  plot first frame with rotation center
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(image_stack[0], cmap="gray")
ax.scatter(rotation_center[1], rotation_center[0], color="red", s=20)
ax.set_title("Rotation center")
ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Perform line-by-line derotation

derotated_stack = derotate_an_image_array_line_by_line(
    image_stack=image_stack,
    rot_deg_line=angles_per_line,
    center=rotation_center,
    blank_pixels_value=blank_pixels_value,
)

# %%
# Visualize the result using a maximum intensity projection

print("Computing max projection for visualization...")
max_proj = np.max(derotated_stack, axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(max_proj, cmap="viridis")
ax.set_title("Maximum Intensity Projection of Derotated Movie")
ax.axis("off")
plt.tight_layout()
plt.show()
