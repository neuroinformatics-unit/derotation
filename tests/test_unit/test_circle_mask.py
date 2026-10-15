"""Unit tests for FullPipeline.add_circle_mask."""

import numpy as np
import pytest

from derotation.analysis.full_derotation_pipeline import FullPipeline


@pytest.mark.parametrize(
    "height,width,diameter",
    [
        (64, 64, 64),
        (64, 64, 50),
        (64, 64, 51),  # odd difference between the size and the diameter
        (40, 64, 40),
        (64, 40, 40),
        (41, 64, 40),
    ],
)
def test_add_circle_mask_keeps_the_frame_shape(height, width, diameter):
    image_stack = np.ones((2, height, width))
    masked = FullPipeline.add_circle_mask(image_stack, diameter)
    assert masked.shape == image_stack.shape


def test_add_circle_mask_is_centered_on_non_square_frames():
    height, width = 40, 64
    image_stack = np.ones((1, height, width))
    # the minimum, inside the cropped square, fills the masked area
    image_stack[0, 0, width // 2] = 0
    masked = FullPipeline.add_circle_mask(image_stack, height)[0]

    kept = np.argwhere(masked == 1)
    center_y, center_x = kept.mean(axis=0)
    assert abs(center_y - (height - 1) / 2) < 1
    assert abs(center_x - (width - 1) / 2) < 1
    # the circle touches the top and the bottom but not the sides
    assert kept[:, 0].min() <= 1 and kept[:, 0].max() >= height - 2
    assert kept[:, 1].min() > 10 and kept[:, 1].max() < width - 10
