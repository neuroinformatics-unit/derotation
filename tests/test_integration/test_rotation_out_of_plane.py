from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from derotation.analysis.fit_ellipse import (
    fit_ellipse_to_points,
    plot_ellipse_fit_and_centers,
)
from derotation.simulate.line_scanning_microscope import Rotator
from derotation.simulate.synthetic_data import SyntheticData


def rotate_image_stack(
    plane_angle: float = 0,
    num_frames: int = 50,
    pad: int = 20,
    orientation: float = 0,
) -> Tuple[np.ndarray, np.ndarray, Rotator, int]:
    """
    Rotates an image stack with a specified plane angle and
    optional orientation.

    Parameters
    ----------
    plane_angle : float, optional
        The angle of the rotation plane in degrees.
    num_frames : int
        Number of frames in the image stack.
    pad : int
        Padding size to apply to the sample image.
    orientation : float, optional
        Orientation of the rotation plane in degrees, by default None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Rotator, int]
        Original image stack, rotated image stack, Rotator instance,
        and number of frames.
    """

    s_data = SyntheticData(
        radius=1,
        second_cell=False,
        pad=pad,
        background_value=80,
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
    return image_stack, rotated_image_stack, rotator, num_frames


@pytest.mark.parametrize(
    "plane_angle,exp_orientation",
    [
        (0, 0),
        (15, 0),
        (25, 0),
        (25, 45),
        (25, 90),
        (40, 0),
        (40, 45),
        (40, 90),
    ],
)
def test_max_projection(
    plane_angle: float,
    exp_orientation: float,
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
    plane_angle : float
        The angle of the rotation plane in degrees.
    exp_orientation : float
        Expected orientation of the rotation plane, by default None.
    atol : int, optional
        Allowed tolerance for orientation, by default 15.
    """
    _, rotated_image_stack, *_ = rotate_image_stack(
        plane_angle=plane_angle, orientation=exp_orientation
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

    xc, yc, a, b, orientation = fit_ellipse_to_points(
        centers, pixels_in_row=rotated_image_stack.shape[1]
    )

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
            assert False, (
                f"Major and minor axes should be close,instead got {a} and {b}"
            )
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
