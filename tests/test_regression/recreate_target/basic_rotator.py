from PIL import Image
from shared import (
    get_increasing_angles,
    get_static_video,
    regenerate_rotator_images_for_testing,
)

if __name__ == "__main__":
    # Get the static "video" of gray-striped squares and monotonic angles
    static_video = get_static_video()
    angles = get_increasing_angles(static_video)
    image = Image.fromarray(static_video[0].astype("uint8"))
    image.save("tests/test_regression/images/rotator/original_image.png")

    # Regenerate rotated images as if they were acquired by a line
    # scanning microscope
    regenerate_rotator_images_for_testing(static_video, angles)
