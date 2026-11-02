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


def test_derotate_cropped_image_matches_cropped_square_result():
    """Cropping the frames in x and moving the center by the same amount
    gives the cropped square result, wherever the cropped frames have data.
    """
    rng = np.random.default_rng(0)
    size, crop = 60, 12
    num_frames = 2
    image_stack = rng.integers(1, 255, (num_frames, size, size)).astype(
        np.float64
    )
    rot_deg_line = np.linspace(0, 90, num_frames * size)
    center = (size // 2, size // 2)

    square = derotate_an_image_array_line_by_line(
        image_stack, rot_deg_line, center=center, blank_pixels_value=0
    )
    cropped = derotate_an_image_array_line_by_line(
        image_stack[:, :, crop:-crop],
        rot_deg_line,
        center=(center[0] - crop, center[1]),
        blank_pixels_value=0,
    )

    expected = square[:, :, crop:-crop]
    assert cropped.shape == expected.shape
    filled = cropped != 0
    assert filled.mean() > 0.6
    assert np.mean(cropped[filled] == expected[filled]) > 0.99

def test_derotation_does_not_leave_blank_holes_inside_image():
    """Derotation should not leave significant blank gaps in the image."""
    num_frames, size = 180, 256
    image_stack = np.full(
        (num_frames, size, size), 255, dtype=np.uint8
    )
    rot_deg_line = np.linspace(0, 90, num_frames * size)

    derotated = derotate_an_image_array_line_by_line(
        image_stack,
        rot_deg_line,
        blank_pixels_value=0,
    )

    rows, cols = np.ogrid[:size, :size]
    inside = (
        (rows - (size - 1) / 2) ** 2
        + (cols - (size - 1) / 2) ** 2
        <= (size / 2 - 2) ** 2
    )

    blank_percent = np.array([
        100 * np.mean(frame[inside] == 0)
        for frame in derotated
    ])

    assert blank_percent.max() < 1.0
