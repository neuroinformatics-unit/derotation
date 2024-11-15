import numpy as np
from shared import (
    NUMBER_OF_FRAMES,
    load_rotated_images,
    regenerate_derotated_images_for_testing,
)

if __name__ == "__main__":
    # Define paths and parameters

    # Angles must match those used for the original rotation
    n_total_lines = (
        100 * NUMBER_OF_FRAMES
    )  # Assuming 100 lines per frame and `stack_len` frames
    angles = np.arange(n_total_lines)

    # Load rotated images
    rotated_image_stack = load_rotated_images()

    # Regenerate derotated images
    regenerate_derotated_images_for_testing(rotated_image_stack, angles)
