from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from derotation.analysis.fit_ellipse import (
    fit_ellipse_to_points,
    plot_ellipse_fit_and_centers,
)
from derotation.simulate.line_scanning_microscope import Rotator
from tests.test_integration.test_derotation_with_simulated_data import (
    create_image_stack,
    create_rotation_angles,
    create_sample_image_with_two_cells,
)


def rotate_image_stack(
    plane_angle: int = 0,
    num_frames: int = 50,
    pad: int = 20,
    orientation: Optional[int] = None,
    center_of_rotation_diff: Tuple[int, int] = (0, 0),
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
    cells = create_sample_image_with_two_cells(
        lines_per_frame=100, second_cell=False, radius=1
    )
    cells = np.pad(cells, ((pad, pad), (pad, pad)), mode="constant")
    cells[cells == 0] = 80

    image_stack = create_image_stack(cells, num_frames=num_frames)
    _, angles = create_rotation_angles(image_stack.shape)

    n_lines = image_stack.shape[1]
    if center_of_rotation_diff is None:
        center = (n_lines // 2, n_lines // 2)
    else:
        center = (
            n_lines // 2 + center_of_rotation_diff[0],
            n_lines // 2 + center_of_rotation_diff[1],
        )

    rotator = Rotator(
        angles,
        image_stack,
        rotation_plane_angle=plane_angle,
        blank_pixel_val=0,
        rotation_plane_orientation=orientation,
        center=center,
    )
    rotated_image_stack = rotator.rotate_by_line()
    return image_stack, rotated_image_stack, rotator, num_frames


@pytest.mark.parametrize(
    "plane_angle,exp_orientation,center_of_rotation_diff",
    [
        (0, None, None),
        (15, None, None),
        (25, None, None),
        (25, 45, None),
        (25, 90, None),
        (25, 45, (-4, 2)),
        (40, None, None),
        (40, 45, None),
        (40, 90, None),
    ],
)
def test_max_projection(
    plane_angle: int,
    exp_orientation: Optional[int],
    center_of_rotation_diff: Tuple[int, int],
    atol: int = 15,
) -> None:
    """
    Tests if the max projection of an image stack fits an ellipse,
    and verify orientation and major/minor axes are close to expected values:
    - if plane_angle is 0, the major and minor axes should be close to each
      other;
    - if exp_orientation is not None, the orientation should be close to
      it.
    - if exp_orientation is None, the orientation should be close to 0.

    Giving by default an allowed tolerance of 15 degrees as I care about
    the test tracking the general shape of the ellipse, not the exact values.

    Parameters
    ----------
    plane_angle : int
        The angle of the rotation plane in degrees.
    exp_orientation : int, optional
        Expected orientation of the rotation plane, by default None.
    atol : int, optional
        Allowed tolerance for orientation, by default 5.
    """
    _, rotated_image_stack, *_ = rotate_image_stack(
        plane_angle=plane_angle,
        orientation=exp_orientation,
        center_of_rotation_diff=center_of_rotation_diff,
    )

    # get the location of the brightest pixel in any frame as a center
    centers = np.array(
        [
            np.unravel_index(np.argmax(frame), frame.shape)
            for frame in rotated_image_stack
        ]
    )
    #  invert centers to (x, y) format
    centers = np.array([(x, y) for y, x in centers])

    xc, yc, a, b, orientation = fit_ellipse_to_points(centers)

    if exp_orientation is None and plane_angle == 0:
        #  lower tolerance for the major and minor axes
        if not np.allclose(a, b, atol=atol - 5):
            plot_ellipse_fit_and_centers(
                image_stack=rotated_image_stack,
                centers=centers,
                center_x=xc,
                center_y=yc,
                a=a,
                b=b,
                theta=orientation,
                debug_plots_folder=Path("debug/"),
                saving_name=f"ellipse_fit_{plane_angle}_{exp_orientation}.png",
            )
            assert (
                False
            ), f"Major and minor axes should be close, instead got {a} and {b}"
    elif exp_orientation is not None:
        #  Major axis orientation in clockwise direction as radians.
        exp_orientation = np.deg2rad(exp_orientation)
        if not np.allclose(orientation, exp_orientation, atol=atol):
            plot_ellipse_fit_and_centers(
                image_stack=rotated_image_stack,
                centers=centers,
                center_x=xc,
                center_y=yc,
                a=a,
                b=b,
                theta=orientation,
                debug_plots_folder=Path("debug/"),
                saving_name=f"ellipse_fit_{plane_angle}_{exp_orientation}.png",
            )
            assert False, (
                f"Orientation should be close to {exp_orientation}, "
                f"instead got {orientation}"
            )
    else:
        # check that the major axis (a) differs from the minor axis (b)
        # as calculated with cosine
        decrease = np.cos(np.deg2rad(plane_angle))
        expected_b = a - a * decrease
        #  lower tolerance for the major and minor axes
        if np.allclose(b, expected_b, atol=atol - 5):
            plot_ellipse_fit_and_centers(
                image_stack=rotated_image_stack,
                centers=centers,
                center_x=xc,
                center_y=yc,
                a=a,
                b=b,
                theta=orientation,
                debug_plots_folder=Path("debug/"),
                saving_name=f"ellipse_fit_{plane_angle}_{exp_orientation}.png",
            )
            assert False, (
                f"Difference between major and minor axes should be close to "
                f"{expected_b}, instead got {b}"
            )


import matplotlib.pyplot as plt


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


def test_make_images():
    image_stack, rotated_image_stack, rotator, num_frames = rotate_image_stack(
        plane_angle=25,
        num_frames=50,
        pad=20,
        orientation=45,
        center_of_rotation_diff=(-10, 10),
    )
    make_plot(
        image_stack,
        rotated_image_stack,
        rotator,
        num_frames,
        title="rotation_out_of_plane",
    )