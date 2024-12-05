from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_ellipse
from test_finding_center_of_rotation_by_joining_two_pipelines import (
    create_image_stack,
    create_rotation_angles,
    create_sample_image_with_two_cells,
)

from derotation.simulate.line_scanning_microscope import Rotator


def rotate_image_stack(
    plane_angle: int,
    num_frames: int,
    pad: int,
    orientation: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Rotator, int]:
    """
    Rotates an image stack with a specified plane angle and
    optional orientation.

    Parameters
    ----------
    plane_angle : int
        The angle of the rotation plane in degrees.
    num_frames : int
        Number of frames in the image stack.
    pad : int
        Padding size to apply to the sample image.
    orientation : int, optional
        Orientation of the rotation plane in degrees, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Rotator, int]
        Original image stack, rotated image stack, Rotator instance,
        and number of frames.
    """
    cells = create_sample_image_with_two_cells(lines_per_frame=100)
    cells = np.pad(cells, ((pad, pad), (pad, pad)), mode="constant")
    cells[cells == 0] = 80

    image_stack = create_image_stack(cells, num_frames=num_frames)
    _, angles = create_rotation_angles(image_stack.shape)

    rotator = Rotator(
        angles,
        image_stack,
        rotation_plane_angle=plane_angle,
        blank_pixel_val=0,
        rotation_plane_orientation=orientation,
    )
    rotated_image_stack = rotator.rotate_by_line()
    return image_stack, rotated_image_stack, rotator, num_frames


def make_plot(
    image_stack: np.ndarray,
    rotated_image_stack: np.ndarray,
    rotator: Rotator,
    num_frames: int,
    title: str = "",
) -> None:
    """
    Plots the original and rotated image stacks.

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


def test_max_projection(
    rotated_image_stack: np.ndarray,
    expected_orientation: Optional[int] = None,
    atol: int = 5,
) -> None:
    """
    Tests the max projection for an image stack to validate rotation.

    Parameters
    ----------
    rotated_image_stack : np.ndarray
        Rotated image stack.
    expected_orientation : int, optional
        Expected orientation of the rotation plane, by default None.
    atol : int, optional
        Allowed tolerance for orientation, by default 5.
    """
    max_projection = rotated_image_stack.max(axis=0)
    edges = canny(max_projection, sigma=7, low_threshold=35, high_threshold=50)
    result = hough_ellipse(
        edges, accuracy=10, threshold=100, min_size=30, max_size=100
    )
    result.sort(order="accumulator")

    best = result[-1]
    yc, xc, a, b, o = [int(best[i]) for i in range(1, len(best))]

    if expected_orientation is not None:
        assert np.allclose(
            o, expected_orientation, atol=atol
        ), f"Orientation should be close to {expected_orientation}"
    else:
        assert o < 5, "Orientation should be close to 0"


if __name__ == "__main__":
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
