import numpy as np
from assertions import comapre_images
from PIL import Image

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from tests.test_regression.recreate_target.shared import (
    DEROTATED_IMAGES_PATH,
    load_rotated_images,
)


def test_derotator_by_line(angles):
    """Test that rotating and derotating the image
    stack restores the original images."""
    rotated_images = load_rotated_images()

    derotated_image_stack = derotate_an_image_array_line_by_line(
        rotated_images, angles
    )

    # Check each derotated frame against precomputed expected images
    for i, derotated_frame in enumerate(derotated_image_stack):
        target_image = Image.open(
            DEROTATED_IMAGES_PATH / f"derotated_frame_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        comapre_images(
            i, derotated_frame, target_image, 1, DEROTATED_IMAGES_PATH, 10
        )
