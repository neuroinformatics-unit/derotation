from pathlib import Path

import numpy as np
from PIL import Image

from derotation.simulate.basic_rotator import Rotator


def rotate_images(image_stack, angles):
    """Rotate the image stack using the Rotator."""
    rotator = Rotator(angles, image_stack)
    return rotator.rotate_by_line()


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
    gray_values = [i % 5 * 60 + 15 for i in range(100)]
    for i in range(100):
        image[i] = gray_values[i]
    image[:20] = 0
    image[-20:] = 0
    image[:, :20] = 0
    image[:, -20:] = 0

    # Creating a stack and angles
    stack_len = 3
    stack = np.array([image for _ in range(stack_len)])
    #  save image
    image = Image.fromarray(image.astype("uint8"))
    image.save("tests/test_regression/images/rotator/original_image.png")

    n_total_lines = stack.shape[0] * stack.shape[1]
    _angles = np.arange(n_total_lines)

    # Regenerate images
    regenerate_rotator_images_for_testing(stack, _angles)
