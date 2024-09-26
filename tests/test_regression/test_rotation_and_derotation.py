from pathlib import Path

import numpy as np
from PIL import Image

from derotation.derotate_by_line import derotate_an_image_array_line_by_line


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


def regenerate_derotated_images(rotated_image_stack, angles, output_directory):
    """Regenerate only the derotated images from rotated images."""
    # Derotate image stack
    derotated_image_stack = derotate_an_image_array_line_by_line(
        rotated_image_stack, angles
    )

    # Save derotated images
    # Create output directory if it does not exist
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    for i, derotated_frame in enumerate(derotated_image_stack):
        derotated_image = Image.fromarray(derotated_frame.astype("uint8"))
        derotated_image.save(
            Path(output_directory) / f"derotated_frame_{i + 1}.png"
        )


def load_rotated_images(directory, len_stack):
    """Load precomputed rotated images from a directory."""
    rotated_image_stack = []
    for i in range(1, len_stack + 1):
        image_path = Path(directory) / f"rotated_frame_{i}.png"
        rotated_image = Image.open(image_path).convert("L")
        rotated_image_stack.append(np.array(rotated_image))
    return np.array(rotated_image_stack)


if __name__ == "__main__":
    # Define paths and parameters
    rotated_images_directory = "tests/test_regression/images/rotator"
    derotated_images_directory = (
        "tests/test_regression/images/rotator_derotator"
    )
    stack_len = 3

    # Angles must match those used for the original rotation
    n_total_lines = (
        100 * stack_len
    )  # Assuming 100 lines per frame and `stack_len` frames
    angles = np.arange(n_total_lines)

    # Load rotated images
    rotated_image_stack = load_rotated_images(
        rotated_images_directory, stack_len
    )

    # Regenerate derotated images
    regenerate_derotated_images(
        rotated_image_stack, angles, derotated_images_directory
    )
