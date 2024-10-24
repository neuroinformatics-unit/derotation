from shared import (
    get_image_stack_and_angles,
    load_rotated_images,
    regenerate_derotated_images_for_testing,
    regenerate_rotator_images_for_testing,
)

if __name__ == "__main__":
    # Set up an image stack and angles
    stack, angles = get_image_stack_and_angles()

    # Regenerate rotated images for custom center (40, 40)
    regenerate_rotator_images_for_testing(stack, angles, center=(40, 40))

    # Define paths for saving derotated images
    _rotated_images_directory = "tests/test_regression/images/rotator"
    derotated_images_directory = (
        "tests/test_regression/images/rotator_derotator"
    )

    # Load rotated images for center (40, 40)
    rotated_image_stack = load_rotated_images(
        _rotated_images_directory, stack.shape[0], center=(40, 40)
    )

    # Regenerate derotated images for center (40, 40)
    regenerate_derotated_images_for_testing(
        rotated_image_stack,
        angles,
        output_directory="tests/test_regression/images/rotator_derotator",
        center=(40, 40),
    )
