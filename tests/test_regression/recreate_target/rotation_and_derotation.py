import numpy as np
from shared import load_rotated_images, regenerate_derotated_images_for_testing

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
    regenerate_derotated_images_for_testing(
        rotated_image_stack, angles, derotated_images_directory
    )
