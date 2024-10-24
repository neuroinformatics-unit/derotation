from PIL import Image
from shared import (
    get_image_stack_and_angles,
    regenerate_rotator_images_for_testing,
    square_with_gray_stripes_in_black_background,
)

if __name__ == "__main__":
    image = square_with_gray_stripes_in_black_background()
    image = Image.fromarray(image.astype("uint8"))
    image.save("tests/test_regression/images/rotator/original_image.png")
    stack, angles = get_image_stack_and_angles()

    # Regenerate images
    regenerate_rotator_images_for_testing(stack, angles)
