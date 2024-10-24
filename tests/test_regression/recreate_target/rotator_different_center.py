from shared import (
    get_image_stack_and_angles,
    regenerate_rotator_images_for_testing,
)

if __name__ == "__main__":
    # Set up an image stack and angles
    stack, angles = get_image_stack_and_angles()

    # Regenerate images for default center and custom center
    regenerate_rotator_images_for_testing(stack, angles, center=(40, 40))
