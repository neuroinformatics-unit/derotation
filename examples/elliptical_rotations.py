"""
Simulate out-of-plane rotation during line-scanning acquisition
===============================================================

In this tutorial, we simulate how image stacks appear when the scanning
happens across a plane that is *not aligned* with the imaging plane.

We will:
    - Generate synthetic 2D frames with circular cell structures.
    - Simulate line-by-line rotations with varying angles.
    - Visualise how the appearance of the image is distorted when the rotation
      plane is tilted or oriented differently.
    - Display all frames and a max projection.

"""

# %%
# Imports
# -------
from pathlib import Path

import matplotlib.pyplot as plt

from derotation.simulate.line_scanning_microscope import Rotator
from derotation.simulate.synthetic_data import SyntheticData

# %%
# Define rotation + plotting functions
# ------------------------------------


def rotate_image_stack(
    plane_angle: float = 0,
    pad: int = 20,
    orientation: float = 0,
):
    """
    Create and rotate a synthetic image stack using the specified
    rotation parameters.
    """
    s_data = SyntheticData(
        radius=1,
        second_cell=False,
        pad=pad,
        background_value=80,
        num_frames=50,
    )
    s_data.image = s_data.create_sample_image_with_cells()
    image_stack = s_data.create_image_stack()
    _, angles = s_data.create_rotation_angles(image_stack.shape)

    rotator = Rotator(
        angles,
        image_stack,
        rotation_plane_angle=plane_angle,
        blank_pixel_val=0,
        rotation_plane_orientation=orientation,
    )
    rotated_image_stack = rotator.rotate_by_line()

    return image_stack, rotated_image_stack, rotator, image_stack.shape[0]


def make_plot(
    image_stack,
    rotated_image_stack,
    rotator,
    num_frames,
    title="",
):
    """
    Plot all frames of the rotated stack and their associated angles.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    max_proj = rotated_image_stack.max(axis=0)
    ax.imshow(max_proj, cmap="gray", vmin=0, vmax=255)
    ax.plot(
        max_proj.shape[1] / 2,
        max_proj.shape[0] / 2,
        "rx",
        markersize=10,
    )
    ax.set_title("Max projection")
    ax.axis("off")

    plt.tight_layout()
    plt.suptitle(title, fontsize=14)
    plt.subplots_adjust(top=0.92)
    plt.show()


# %%
# Create output folder (optional)
Path("debug").mkdir(exist_ok=True)

# %%
# Example 1 – rotation out of imaging plane
# -----------------------------------------
# Here we simulate a 25° tilt in the rotation plane, with no orientation
# shift. This simulates a case where the imaging scan plane is not aligned
# with the rotation axis.

image_stack, rotated_image_stack, rotator, num_frames = rotate_image_stack(
    plane_angle=25, pad=20
)

print("Rotation plane angle: 25°")
print("Rotation orientation: 0°")

make_plot(
    image_stack,
    rotated_image_stack,
    rotator,
    num_frames,
    title="Out-of-plane rotation (25° tilt)",
)

# %%
# Example 2 – rotation + in-plane orientation
# -------------------------------------------
# Now we also add a 45° orientation to the rotation plane, so it's both tilted
# and diagonally oriented relative to the image.

image_stack, rotated_image_stack, rotator, num_frames = rotate_image_stack(
    plane_angle=25, pad=20, orientation=45
)

print("Rotation plane angle: 25°")
print("Rotation orientation: 45°")

make_plot(
    image_stack,
    rotated_image_stack,
    rotator,
    num_frames,
    title="Tilted + Oriented Rotation Plane (25°, 45°)",
)

# %%
# Conclusion
# ----------
# This simulation helps us visualise how image distortions appear during
# line-scanning acquisition when the imaging plane is misaligned with the
# physical rotation plane. The observed distortions depend on both the *angle*
# of the rotation plane and its *orientation* in space.
