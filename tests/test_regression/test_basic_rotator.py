import numpy as np
from assertions import comapre_images
from PIL import Image

from tests.test_regression.recreate_target.shared import (
    ROTATED_IMAGES_PATH,
    rotate_images,
)


def test_rotator_by_line(image_stack, angles):
    """Test that the Rotator correctly rotates the image stack by line."""
    # Perform rotation
    rotated_image_stack = rotate_images(image_stack, angles)

    # Check each rotated frame against precomputed expected images
    for i, rotated_frame in enumerate(rotated_image_stack):
        target_image = Image.open(
            ROTATED_IMAGES_PATH / f"rotated_frame_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        comapre_images(
            i, rotated_frame, target_image, 1, ROTATED_IMAGES_PATH, 0
        )
