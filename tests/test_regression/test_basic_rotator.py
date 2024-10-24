import numpy as np
from PIL import Image

from tests.test_regression.recreate_target.shared import rotate_images


def test_rotator_by_line(image_stack, angles, len_stack):
    """Test that the Rotator correctly rotates the image stack by line."""
    # Perform rotation
    rotated_image_stack = rotate_images(image_stack, angles)

    # Check each rotated frame against precomputed expected images
    for i, rotated_frame in enumerate(rotated_image_stack):
        target_image = Image.open(
            f"tests/test_regression/images/rotator/rotated_frame_{i + 1}.png"
        )
        target_image = np.array(target_image.convert("L"))

        # Compare each frame against the precomputed target image
        try:
            assert np.allclose(
                rotated_frame, target_image, atol=1
            ), f"Failed for frame {i + 1}"
        except AssertionError:
            #  print where it is different
            diff = np.abs(rotated_frame - target_image)

            #  which indexes are different
            indexes = np.where(diff > 1)
            #  save wrong image
            rotated_image = Image.fromarray(rotated_frame.astype("uint8"))
            rotated_image.save(
                "tests/test_regression/images/rotator/"
                + f"rotated_frame_{i + 1}_wrong.png"
            )

            assert False, (
                "Index where it is different: "
                + f"{indexes}, Total: {len(indexes[0])}"
            )
