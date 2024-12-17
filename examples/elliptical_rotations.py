#  Visualize the acquisition of a movie where the rotation axis is not
#  aligned with the image plane.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from derotation.simulate.line_scanning_microscope import Rotator
from tests.test_integration.test_rotation_out_of_plane import (
    rotate_image_stack,
)


def make_plot(
    image_stack: np.ndarray,
    rotated_image_stack: np.ndarray,
    rotator: Rotator,
    num_frames: int,
    title: str = "",
) -> None:
    """
    Make figure with rotated image stacks frame by frame.

    Parameters
    ----------
    image_stack : np.ndarray
        Original image stack.
    rotated_image_stack : np.ndarray
        Rotated image stack.
    rotator : Rotator
        Rotator instance containing rotation metadata.
    num_frames : int
        Number of frames in the image stack.
    title : str, optional
        Title for the saved plot, by default "".
    """
    row_n = 5
    fig, ax = plt.subplots(row_n, num_frames // row_n + 1, figsize=(40, 25))

    ax[0, 0].imshow(image_stack[0], cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Original image")
    ax[0, 0].axis("off")

    for n in range(1, len(rotated_image_stack) + 1):
        row = n % row_n
        col = n // row_n
        ax[row, col].imshow(
            rotated_image_stack[n - 1], cmap="gray", vmin=0, vmax=255
        )
        ax[row, col].set_title(n)

        angles = rotator.angles[n - 1]
        angle_range = f"{angles.min():.0f}-{angles.max():.0f}"
        ax[row, col].text(
            0.5,
            0.9,
            angle_range,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax[row, col].transAxes,
            color="white",
        )

    ax[row_n - 1, num_frames // row_n].imshow(
        rotated_image_stack.max(axis=0), cmap="gray"
    )
    ax[row_n - 1, num_frames // row_n].plot(
        rotated_image_stack.shape[2] / 2,
        rotated_image_stack.shape[1] / 2,
        "rx",
        markersize=10,
    )
    ax[row_n - 1, num_frames // row_n].set_title("Max projection")

    for a in ax.ravel():
        a.axis("off")

    plt.savefig(f"debug/{title}.png")


Path("debug").mkdir(exist_ok=True)

image_stack, rotated_image_stack, rotator, num_frames = rotate_image_stack(
    plane_angle=25, num_frames=50, pad=20
)
make_plot(
    image_stack,
    rotated_image_stack,
    rotator,
    num_frames,
    title="rotation_out_of_plane",
)

image_stack, rotated_image_stack, rotator, num_frames = rotate_image_stack(
    plane_angle=25, num_frames=50, pad=20, orientation=45
)
make_plot(
    image_stack,
    rotated_image_stack,
    rotator,
    num_frames,
    title="rotation_out_of_plane_plus_orientation",
)
