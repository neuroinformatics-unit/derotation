"""
Find center of rotation with simulated data
================================================================

This tutorial use the synthetic data class to show how it can be possible
to use an incremental rotation movie to find the center of rotation,
rotation plane angle and rotation plane orientation and then derotate
a movie with any angle pattern.

What happens behind the scenes:
    - Generate a synthetic image with two cells and two angle patterns,
      one for incremental pipeline (in steps) and one for testing (a
      sinusoidal pattern).
    - Use the Rotator class to derotate the movie with the incremental angle
      pattern and the sinusoidal angle pattern.
    - Find the center of rotation by fitting an ellipse to the largest blob
      for different angle positions.
    - Derotate the movie with the sinusoidal angle pattern using the
      center of rotation and ellipse parameters found in the previous step.

We are going to inspect the plots generated during the pipeline to see
how the algorithm works and how the center of rotation is found.

This serves only as an illustrative example. To understand how to find the
center of rotation with real data, please refer to the
`User guide <../user_guide/key_concepts.html>`_.
"""

# %%
# Imports
# -------
from pathlib import Path

import matplotlib.pyplot as plt

from derotation.simulate.synthetic_data import SyntheticData

# %%
# Define parameters for the synthetic simulation
# ----------------------------------------------

rotation_plane_angle = 10  # degrees
rotation_plane_orientation = 30  # degrees
center_of_rotation_offset = (7, -7)  # pixels

# %%
# Run full pipeline
# -----------------
# This will generate simulated rotated data and run an incremental search
# to find the optimal center of rotation.
# All intermediate plots are saved in the "debug" folder.

s_data = SyntheticData(
    radius=2,
    center_of_rotation_offset=center_of_rotation_offset,
    rotation_plane_angle=rotation_plane_angle,
    rotation_plane_orientation=rotation_plane_orientation,
    num_frames=50,
    pad=20,
    background_value=80,
    plots=True,
)

s_data.integration_pipeline()

# %%
# Display relevant plots
# -----------------------
# Let's display some of the plots generated during the pipeline.

debug_folder = Path("debug")
debug_images = sorted(debug_folder.glob("*.png"))


def get_image_path(name):
    return [img_path for img_path in debug_images if name in img_path.name][0]


def show_image(path):
    img = plt.imread(path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# %%
# The synthetic image with two cells that was generated
show_image(get_image_path("image_"))

# %%
#  Rotation angles for incremental pipeline and sinusoidal pattern
show_image(get_image_path("rotation_angles"))

# %%
# Fit an ellipse to the largest blob for different angle positions
show_image(get_image_path("ellipse_fit"))

print(f"Estimated center of rotation: {s_data.fitted_center}")
print(f"Estimated rotation plane angle: {s_data.rotation_plane_angle}")
print(
    "Estimated rotation plane orientation: "
    f"{s_data.rotation_plane_orientation}"
)

# %%
# Derotated movie with the sinusoidal angle pattern using the information
# found in the previous step
show_image(get_image_path("derotated_sinusoidal"))

# %%
# Plot mean projection of the derotated movie as a check of the derotation
# quality
show_image(get_image_path("mean_projection"))
