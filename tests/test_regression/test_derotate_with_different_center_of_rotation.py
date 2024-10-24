from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from tests.test_regression.recreate_target.derotate_different_center import (
    load_rotated_images,
)


@pytest.fixture
def rotated_images_directory():
    return "tests/test_regression/images/rotator"


@pytest.mark.parametrize("center", [(40, 40)])
def test_derotator_by_line_with_center(
    rotated_images_directory, angles, center
):
    """Test that rotating and derotating the image stack
    restores the original images for a given center."""
    # Load rotated images
    rotated_images = load_rotated_images(
        rotated_images_directory, 3, center=center
    )

    # Perform derotation with the same center
    derotated_image_stack = derotate_an_image_array_line_by_line(
        rotated_images, angles, center=center
    )

    # Check each derotated frame against precomputed expected images
    center_suffix = f"{center[0]}_{center[1]}"

    errors = 0
    atol = 1
    acceptance_threshold = 10

    for i, derotated_frame in enumerate(derotated_image_stack):
        target_image = Image.open(
            "tests/test_regression/images/rotator_derotator/"
            + f"derotated_frame_{center_suffix}_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        # Compare each frame against the precomputed target image
        if not np.allclose(derotated_frame, target_image, atol=atol):
            differences = np.abs(derotated_frame - target_image)
            errors += differences[differences > atol].size

    # Check if the number of errors is within the acceptance threshold
    if errors > acceptance_threshold:
        # Save the incorrect image
        wrong_image = Image.fromarray(derotated_frame.astype("uint8"))
        wrong_image.save(
            Path("tests/test_regression/images/rotator_derotator")
            / f"wrong_derotated_frame_{center_suffix}_{i + 1}.png"
        )
        assert False, (
            f"More than {acceptance_threshold} errors in derotation,"
            f" in total there were {errors} errors."
        )
