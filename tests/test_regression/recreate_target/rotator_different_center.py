from shared import (
    get_increasing_angles,
    get_static_video,
    regenerate_rotator_images_for_testing,
)

if __name__ == "__main__":
    # Set up an image stack and angles
    static_video = get_static_video()
    angles = get_increasing_angles(static_video)

    # Regenerate images for default center and custom center
    regenerate_rotator_images_for_testing(
        static_video, angles, center=(40, 40)
    )
