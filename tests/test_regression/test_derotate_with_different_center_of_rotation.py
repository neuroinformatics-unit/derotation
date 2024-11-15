import numpy as np
import pytest
from assertions import comapre_images
from PIL import Image

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from tests.test_regression.recreate_target.shared import (
    DEROTATED_IMAGES_PATH,
    load_rotated_images,
)


@pytest.mark.parametrize("center", [(40, 40)])
def test_derotator_by_line_with_center(angles, center):
    """Test that rotating and derotating the image stack
    restores the original images for a given center."""

    # Load rotated images
    rotated_images = load_rotated_images(center)

    # Perform derotation with the same center
    derotated_image_stack = derotate_an_image_array_line_by_line(
        rotated_images, angles, center=center
    )

    # Check each derotated frame against precomputed expected images
    center_suffix = f"{center[0]}_{center[1]}"

    for i, derotated_frame in enumerate(derotated_image_stack):
        target_image = Image.open(
            DEROTATED_IMAGES_PATH
            / f"derotated_frame_{center_suffix}_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        comapre_images
        (
            i,
            derotated_frame,
            target_image,
            1,
            DEROTATED_IMAGES_PATH,
            10,
        )
