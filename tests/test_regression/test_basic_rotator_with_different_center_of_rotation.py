import numpy as np
import pytest
from assertions import compare_images
from PIL import Image

from derotation.simulate.basic_rotator import Rotator
from tests.test_regression.recreate_target.shared import (
    ROTATED_IMAGES_PATH,
    center_formatting,
    rotate_images,
)


def test_center_50_50_is_same_to_None(image_stack, angles):
    """Test that the default center is the same as (50, 50)."""
    rotator_default = Rotator(angles, image_stack)
    rotator_custom = Rotator(angles, image_stack, center=(50, 50))

    default_rotation = rotator_default.rotate_by_line()
    custom_rotation = rotator_custom.rotate_by_line()

    for i in range(len(default_rotation)):
        compare_images(
            i,
            default_rotation[i],
            custom_rotation[i],
            atol=1,
            save_location=ROTATED_IMAGES_PATH,
        )


@pytest.mark.parametrize("center", [(40, 40)])
def test_rotator_by_line(image_stack, angles, center):
    """Test that the Rotator correctly rotates the image stack
    by line for different centers."""

    # Perform rotation
    rotated_image_stack = rotate_images(image_stack, angles, center=center)

    # Check each rotated frame against precomputed expected images
    for i, rotated_frame in enumerate(rotated_image_stack):
        target_image = Image.open(
            ROTATED_IMAGES_PATH
            / f"rotated_frame_{center_formatting(center)}{i + 1}.png"
        )

        target_image = np.array(target_image.convert("L"))

        # Compare each frame against the precomputed target image
        compare_images(
            i,
            rotated_frame,
            target_image,
            atol=1,
            save_location=ROTATED_IMAGES_PATH,
        )
