from pathlib import Path

import numpy as np
from PIL import Image

from derotation.simulate.basic_rotator import Rotator


def rotate_images(image_stack, angles, center=None):
    """Rotate the image stack using the Rotator."""
    rotator = Rotator(angles, image_stack, center=center)
    return rotator.rotate_by_line()


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
