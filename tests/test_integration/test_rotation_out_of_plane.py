from typing import Optional, Tuple

import numpy as np
import pytest
from matplotlib import pyplot as plt
from skimage.draw import ellipse_perimeter
from skimage.feature import canny
from skimage.transform import hough_ellipse

from derotation.simulate.line_scanning_microscope import Rotator
from tests.test_integration.test_finding_center_of_rotation_by_joining_two_pipelines import (
    create_image_stack,
    create_rotation_angles,
    create_sample_image_with_two_cells,
)


def rotate_image_stack(
    plane_angle: int = 0,
    num_frames: int = 50,
    pad: int = 20,
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


@pytest.mark.parametrize(
    "plane_angle,expected_orientation",
    [
        (0, None),
        (15, None),
        (25, None),
        (25, 45),
        (25, 90),
        (25, 135),
        (40, None),
        (40, 45),
        (40, 90),
        (40, 135),
    ],
)
def test_max_projection(
    plane_angle: int,
    expected_orientation: Optional[int],
    atol: int = 15,
) -> None:
    """
    Tests if the max projection of an image stack fits an ellipse,
    and verify orientation and major/minor axes are close to expected values:
    - if plane_angle is 0, the major and minor axes should be close to each other;
    - if expected_orientation is not None, the orientation should be close to it.
    - if expected_orientation is None, the orientation should be close to 0.

    Giving by default an allowed tolerance of 15 degrees as I care about
    the test tracking the general shape of the ellipse, not the exact values.

    Parameters
    ----------
    plane_angle : int
        The angle of the rotation plane in degrees.
    expected_orientation : int, optional
        Expected orientation of the rotation plane, by default None.
    atol : int, optional
        Allowed tolerance for orientation, by default 5.
    """
    _, rotated_image_stack, *_ = rotate_image_stack(
        plane_angle=plane_angle, orientation=expected_orientation
    )

    max_projection = rotated_image_stack.max(axis=0)
    edges = canny(max_projection, sigma=7, low_threshold=30, high_threshold=50)
    result = hough_ellipse(
        edges, accuracy=9, threshold=110, min_size=35, max_size=100
    )
    result.sort(order="accumulator")

    best = result[-1]
    yc, xc, a, b, orientation = [int(best[i]) for i in range(1, len(best))]

    #  orientation: Major axis orientation in clockwise direction as radians.
    #  expected orientation is calculated frm the minor axis, so convert
    orientation = np.abs(np.rad2deg(orientation))

    if expected_orientation is None and plane_angle == 0:
        #  lower tolerance for the major and minor axes
        if not np.allclose(a, b, atol=atol - 5):
            plot_max_projection_with_rotation_out_of_plane(
                max_projection,
                edges,
                yc,
                xc,
                a,
                b,
                orientation,
                f"debug/error_{plane_angle}_{expected_orientation}.png",
            )
            assert (
                False
            ), f"Major and minor axes should be close to each other, instead got {a} and {b}"
    elif expected_orientation is not None:
        if not np.allclose(orientation, expected_orientation, atol=atol):
            plot_max_projection_with_rotation_out_of_plane(
                max_projection,
                edges,
                yc,
                xc,
                a,
                b,
                orientation,
                f"debug/error_{plane_angle}_{expected_orientation}.png",
            )
            assert (
                False
            ), f"Orientation should be close to {expected_orientation}, instead got {orientation}"
    else:
        # check that the major axis (a) differs from the minor axis (b) as calculated with cosine
        decrease = np.cos(np.deg2rad(plane_angle))
        expected_b = a - a * decrease
        #  lower tolerance for the major and minor axes
        if np.allclose(b, expected_b, atol=atol - 5):
            plot_max_projection_with_rotation_out_of_plane(
                max_projection,
                edges,
                yc,
                xc,
                a,
                b,
                orientation,
                f"debug/error_{plane_angle}_{expected_orientation}.png",
            )
            assert (
                False
            ), f"Difference between major and minor axes should be close to {expected_b}, instead got {b}"


def plot_max_projection_with_rotation_out_of_plane(
    max_projection, edges, yc, xc, a, b, o, title
):
    #  plot for debugging purposes

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(max_projection, cmap="gray")
    ax[0].set_title("Max projection")
    ax[0].axis("off")

    #  plot edges
    ax[1].imshow(edges, cmap="gray")
    ax[1].set_title("Edges")
    ax[1].axis("off")

    print(f"Ellipse center: ({yc}, {xc}), a: {a}, b: {b}, orientation: {o}")

    #  plot the ellipse
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation=o)
    empty_image = np.zeros_like(max_projection)
    empty_image[cy, cx] = 255

    #  show ellipse
    ax[2].imshow(empty_image, cmap="gray")
    ax[2].set_title("Ellipse")
    ax[2].axis("off")

    plt.savefig(title)


if __name__ == "__main__":
    #  To be used for debugging purposes
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
