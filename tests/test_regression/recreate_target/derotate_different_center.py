from tests.test_regression.recreate_target.shared import (
    get_increasing_angles,
    get_static_video,
    load_rotated_images,
    regenerate_derotated_images_for_testing,
    regenerate_rotator_images_for_testing,
)

if __name__ == "__main__":
    # Set up an image stack and angles
    static_video = get_static_video()
    angles = get_increasing_angles(static_video)

    # Regenerate rotated images for custom center (40, 40)
    regenerate_rotator_images_for_testing(
        static_video, angles, center=(40, 40)
    )

    # Load rotated images for center (40, 40)
    rotated_image_stack = load_rotated_images(center=(40, 40))

    # Regenerate derotated images for center (40, 40)
    regenerate_derotated_images_for_testing(
        rotated_image_stack,
        angles,
        center=(40, 40),
    )
