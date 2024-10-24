from PIL import Image
from shared import (
    get_image_stack_and_angles,
    regenerate_rotator_images_for_testing,
)

if __name__ == "__main__":
    # Get the static "video" of gray-striped squares and monotonic angles
    stack, angles = get_image_stack_and_angles()
    image = Image.fromarray(stack[0].astype("uint8"))
    image.save("tests/test_regression/images/rotator/original_image.png")

    # Regenerate rotated images as if they were acquired by a line
    # scanning microscope
    regenerate_rotator_images_for_testing(stack, angles)
