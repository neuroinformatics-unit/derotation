from pathlib import Path

import numpy as np
from PIL import Image

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from tests.test_regression.recreate_target.rotation_and_derotation import (
    load_rotated_images,
)


def test_derotator_by_line(angles):
    """Test that rotating and derotating the image
    stack restores the original images."""
    rotated_images = load_rotated_images(
        "tests/test_regression/images/rotator", 3
    )

    derotated_image_stack = derotate_an_image_array_line_by_line(
        rotated_images, angles
    )

    # Check each derotated frame against precomputed expected images
    for i, derotated_frame in enumerate(derotated_image_stack):
        target_image = Image.open(
            "tests/test_regression/images/rotator_derotator/"
            + f"derotated_frame_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        # Compare each frame against the precomputed target image
        try:
            assert np.allclose(
                derotated_frame, target_image, atol=1
            ), f"Failed for frame {i + 1}"
        except AssertionError:
            #  print where it is different
            diff = np.abs(derotated_frame - target_image)

            #  which indexes are different
            indexes = np.where(diff > 1)

            # save wrong image
            wrong_image = Image.fromarray(derotated_frame.astype("uint8"))
            wrong_image.save(
                Path("tests/test_regression/images/rotator_derotator")
                / f"wrong_derotated_frame_{i + 1}.png"
            )

            assert (
                False
            ), f"Index where it is different: {indexes}, Total: {len(indexes)}"
