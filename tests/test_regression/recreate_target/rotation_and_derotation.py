import numpy as np

from tests.test_regression.recreate_target.shared import (
    NUM_LINES,
    NUMBER_OF_FRAMES,
    load_rotated_images,
    regenerate_derotated_images_for_testing,
)

if __name__ == "__main__":
    # Please run basic_rotator.py first to generate the rotated images
    # if they are not already present in the images/rotator/ directory.

    # Angles must match those used for the original rotation
    n_total_lines = NUM_LINES * NUMBER_OF_FRAMES
    angles = np.arange(n_total_lines)

    # Load rotated images
    rotated_image_stack = load_rotated_images()

    # Regenerate derotated images
    regenerate_derotated_images_for_testing(rotated_image_stack, angles)
