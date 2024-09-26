from pathlib import Path

import numpy as np
from PIL import Image

from derotation.simulate.basic_rotator import Rotator


def rotate_images(image_stack, angles):
    """Rotate the image stack using the Rotator."""
    rotator = Rotator(angles, image_stack)
    return rotator.rotate_by_line()


def test_rotator_by_line(image_stack, angles, len_stack):
    """Test that the Rotator correctly rotates the image stack by line."""
    # Perform rotation
    rotated_image_stack = rotate_images(image_stack, angles)

    # Check each rotated frame against precomputed expected images
    for i, rotated_frame in enumerate(rotated_image_stack):
        target_image = Image.open(
            f"tests/test_regression/images/rotator/rotated_frame_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        # Compare each frame against the precomputed target image
        assert np.allclose(
            rotated_frame, target_image, atol=1
        ), f"Failed for frame {i + 1}"


def regenerate_rotator_images_for_testing(image_stack, angles):
    """Regenerate expected rotated images for regression testing."""
    rotated_image_stack = rotate_images(image_stack, angles)
    path = Path("tests/test_regression/images/rotator")
    path.mkdir(parents=True, exist_ok=True)
    # Save rotated images
    for i, rotated_frame in enumerate(rotated_image_stack):
        rotated_image = Image.fromarray(rotated_frame.astype("uint8"))
        rotated_image.save(path / f"rotated_frame_{i + 1}.png")


if __name__ == "__main__":
    image = np.zeros((100, 100))
    for i in range(100):
        image[i] = i
    image[:20] = 0
    image[-20:] = 0
    image[:, :20] = 0
    image[:, -20:] = 0

    # Creating a stack and angles
    stack_len = 3
    stack = np.array([image for _ in range(stack_len)])
    n_total_lines = stack.shape[0] * stack.shape[1]
    _angles = np.arange(n_total_lines)

    # Regenerate images
    regenerate_rotator_images_for_testing(stack, _angles)
