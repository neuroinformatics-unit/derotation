from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from derotation.simulate.basic_rotator import Rotator


def rotate_images(image_stack, angles, center=None):
    """Rotate the image stack using the Rotator."""
    rotator = Rotator(angles, image_stack, center=center)
    return rotator.rotate_by_line()


def test_center_50_50_is_same_to_None(image_stack, angles):
    """Test that the default center is the same as (50, 50)."""
    rotator_default = Rotator(angles, image_stack)
    rotator_custom = Rotator(angles, image_stack, center=(50, 50))

    default_rotation = rotator_default.rotate_by_line()
    custom_rotation = rotator_custom.rotate_by_line()

    for i in range(len(default_rotation)):
        assert np.allclose(default_rotation[i], custom_rotation[i])


@pytest.mark.parametrize("center", [(40, 40)])
def test_rotator_by_line(image_stack, angles, center):
    """Test that the Rotator correctly rotates the image stack
    by line for different centers."""
    # Perform rotation
    rotated_image_stack = rotate_images(image_stack, angles, center=center)

    center_suffix = "default" if center is None else f"{center[0]}_{center[1]}"

    # Check each rotated frame against precomputed expected images
    for i, rotated_frame in enumerate(rotated_image_stack):
        target_image = Image.open(
            "tests/test_regression/images/rotator/"
            + f"rotated_frame_{center_suffix}_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        # Compare each frame against the precomputed target image
        assert np.allclose(
            rotated_frame, target_image, atol=1
        ), f"Failed for frame {i + 1} with center {center}"


def regenerate_rotator_images_for_testing(image_stack, angles, center=None):
    """Regenerate expected rotated images for regression testing."""
    rotated_image_stack = rotate_images(image_stack, angles, center=center)
    center_suffix = "default" if center is None else f"{center[0]}_{center[1]}"
    path = Path("tests/test_regression/images/rotator")
    path.mkdir(parents=True, exist_ok=True)
    # Save rotated images
    for i, rotated_frame in enumerate(rotated_image_stack):
        rotated_image = Image.fromarray(rotated_frame.astype("uint8"))
        rotated_image.save(path / f"rotated_frame_{center_suffix}_{i + 1}.png")


if __name__ == "__main__":
    # Set up an image stack and angles
    image = np.zeros((100, 100))
    gray_values = [i % 5 * 60 + 15 for i in range(100)]
    for i in range(100):
        image[i] = gray_values[i]
    image[:20] = 0
    image[-20:] = 0
    image[:, :20] = 0
    image[:, -20:] = 0

    stack_len = 3
    stack = np.array([image for _ in range(stack_len)])
    n_total_lines = stack.shape[0] * stack.shape[1]
    _angles = np.arange(n_total_lines)

    # Regenerate images for default center and custom center
    regenerate_rotator_images_for_testing(stack, _angles, center=(40, 40))