from pathlib import Path

import numpy as np
from PIL import Image

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
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


def regenerate_derotated_images_for_testing(
    rotated_image_stack, angles, center, output_directory
):
    """Regenerate derotated images for a given center."""
    # Derotate image stack
    derotated_image_stack = derotate_an_image_array_line_by_line(
        rotated_image_stack, angles, center=center
    )

    # Save derotated images
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    center_suffix = f"{center[0]}_{center[1]}"

    for i, derotated_frame in enumerate(derotated_image_stack):
        derotated_image = Image.fromarray(derotated_frame.astype("uint8"))
        derotated_image.save(
            Path(output_directory)
            / f"derotated_frame_{center_suffix}_{i + 1}.png"
        )


def load_rotated_images(directory, len_stack, center=None):
    """Load precomputed rotated images from a directory."""
    center_suffix = "default" if center is None else f"{center[0]}_{center[1]}"
    rotated_image_stack = []
    for i in range(1, len_stack + 1):
        image_path = Path(directory) / f"rotated_frame_{center_suffix}_{i}.png"
        rotated_image = Image.open(image_path).convert("L")
        rotated_image_stack.append(np.array(rotated_image))
    return np.array(rotated_image_stack)


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
    angles = np.arange(n_total_lines)

    # Regenerate rotated images for custom center (40, 40)
    regenerate_rotator_images_for_testing(stack, angles, center=(40, 40))

    # Define paths for saving derotated images
    _rotated_images_directory = "tests/test_regression/images/rotator"
    derotated_images_directory = (
        "tests/test_regression/images/rotator_derotator"
    )

    # Load rotated images for center (40, 40)
    rotated_image_stack = load_rotated_images(
        _rotated_images_directory, stack_len, center=(40, 40)
    )

    # Regenerate derotated images for center (40, 40)
    regenerate_derotated_images_for_testing(
        rotated_image_stack,
        angles,
        center=(40, 40),
        output_directory="tests/test_regression/images/rotator_derotator",
    )
