"""Unit tests for the derotate_by_line module."""

import numpy as np
import pytest

from derotation.derotate_by_line import derotate_an_image_array_line_by_line


@pytest.mark.parametrize(
    "height,width",
    [
        (100, 160),
        (160, 100),
    ],
)
def test_derotate_non_square_image(height: int, width: int):
    """Test line-by-line derotation on non-square images.

    Parameters
    ----------
    height : int
        Number of lines per frame.
    width : int
        Number of pixels per line.
    """
    num_frames = 2
    image_stack = np.full((num_frames, height, width), 255, dtype=np.uint8)
    rot_deg_line = np.linspace(0, 30, num_frames * height)

    derotated = derotate_an_image_array_line_by_line(image_stack, rot_deg_line)

    assert derotated.shape == (num_frames, height, width)
    assert derotated.dtype == image_stack.dtype


def test_derotate_non_square_image_with_center():
    """Test non-square image derotation with an explicit center."""
    height, width = 100, 160
    num_frames = 2
    image_stack = np.full((num_frames, height, width), 255, dtype=np.uint8)
    rot_deg_line = np.linspace(0, 30, num_frames * height)
    center = (80, 50)

    derotated = derotate_an_image_array_line_by_line(
        image_stack, rot_deg_line, center=center
    )

    assert derotated.shape == (num_frames, height, width)


def test_derotate_non_square_image_with_homography():
    """Test non-square image derotation with homography enabled."""
    height, width = 100, 160
    num_frames = 2
    image_stack = np.full((num_frames, height, width), 255, dtype=np.uint8)
    rot_deg_line = np.linspace(0, 30, num_frames * height)

    derotated = derotate_an_image_array_line_by_line(
        image_stack,
        rot_deg_line,
        use_homography=True,
        rotation_plane_angle=10,
        rotation_plane_orientation=15,
    )

    assert derotated.shape == (num_frames, height, width)
