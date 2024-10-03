import copy
from pathlib import Path

import numpy as np
import pytest
from skimage.feature import blob_log

from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)
from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.simulate.basic_rotator import Rotator


def create_test_image(center1=(50, 10), center2=(60, 60)):
    # Initialize a black image of size 100x100
    image = np.zeros((100, 100), dtype=np.uint8)

    # Define the circle's parameters
    radius = 5  # radius of the circle
    white_value = 255  # white color for the circle

    #  add an extra gray circle at the bottom right
    radius2 = 5
    gray_value = 128

    # Draw a white circle in the top center
    y, x = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 <= radius**2
    image[mask] = white_value

    # Draw a gray circle in the bottom right
    mask2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 <= radius2**2
    image[mask2] = gray_value

    return image


def create_image_stack(image, num_frames):
    return np.array([image for _ in range(num_frames)])


def create_rotation_angles(image_stack):
    num_lines_total = image_stack.shape[1] * image_stack.shape[0]
    num_steps = 360 // 10
    incremental_angles = np.arange(0, num_steps * 10, 10)
    incremental_angles = np.repeat(
        incremental_angles, num_lines_total // len(incremental_angles)
    )

    if len(incremental_angles) < num_lines_total:
        incremental_angles = np.concatenate(
            (
                incremental_angles,
                [incremental_angles[-1]]
                * (num_lines_total - len(incremental_angles)),
            )
        )

    max_rotation = 360  # max rotation angle
    num_cycles = 1
    sinusoidal_angles = max_rotation * np.sin(
        np.linspace(0, num_cycles * 2 * np.pi, num_lines_total)
    )

    return incremental_angles.astype("float64"), sinusoidal_angles


def get_center_of_rotation(rotated_stack_incremental, incremental_angles):
    class MockIncrementalPipeline(IncrementalPipeline):
        def __init__(self):
            self.image_stack = rotated_stack_incremental
            self.rot_deg_frame = incremental_angles[
                :: rotated_stack_incremental.shape[1]
            ]
            self.num_frames = rotated_stack_incremental.shape[0]
            self.debugging_plots = True
            self.debug_plots_folder = Path("debug/")

        def calculate_mean_images(self, image_stack: np.ndarray) -> list:
            #  Overwrite original method as it is too bound
            #  to signal coming from a real motor
            angles_subset = copy.deepcopy(self.rot_deg_frame)
            rounded_angles = np.round(angles_subset)

            mean_images = []
            for i in np.arange(10, 360, 10):
                images = image_stack[rounded_angles == i]
                mean_image = np.mean(images, axis=0)

                mean_images.append(mean_image)

            return mean_images

    pipeline = MockIncrementalPipeline()
    center_of_rotation = pipeline.find_center_of_rotation()

    return center_of_rotation


def integration_pipeline(test_image, center_of_rotation_initial, num_frames):
    image_stack = create_image_stack(test_image, num_frames)

    # Generate rotation angles
    incremental_angles, sinusoidal_angles = create_rotation_angles(image_stack)

    # Use the Rotator to create the rotated image stacks
    rotator_incremental = Rotator(
        incremental_angles, image_stack, center_of_rotation_initial
    )
    rotated_stack_incremental = rotator_incremental.rotate_by_line()

    rotator_sinusoidal = Rotator(
        sinusoidal_angles, image_stack, center_of_rotation_initial
    )
    rotated_stack_sinusoidal = rotator_sinusoidal.rotate_by_line()

    center_of_rotation = get_center_of_rotation(
        rotated_stack_incremental, incremental_angles
    )

    derotated_sinusoidal = derotate_an_image_array_line_by_line(
        rotated_stack_sinusoidal,
        sinusoidal_angles,
        center=center_of_rotation,
    )

    return derotated_sinusoidal


#  parametrize the test
@pytest.mark.parametrize("center_of_rotation_initial", [(44, 51), (51, 44)])
def test_blob_detection_on_derotated_stack(center_of_rotation_initial):
    center_1 = (50, 10)
    center_2 = (60, 60)
    num_frames = 100

    test_image = create_test_image(center1=center_1, center2=center_2)
    derotated_sinusoidal = integration_pipeline(
        test_image, center_of_rotation_initial, num_frames
    )

    blobs = [blob_log(img) for img in derotated_sinusoidal]

    #  for every frame, place first the blob with the smallest x value
    blobs = [sorted(blob, key=lambda x: x[1]) for blob in blobs]

    # compare the first and second blob to the expected values
    errors = 0
    atol = 6  # 5 pixels tolerance for the center of the blobs
    for blob in blobs:
        if not np.allclose(blob[0][:2][::-1], center_1, atol=atol):
            errors += 1
        if len(blob) > 1 and not np.allclose(
            blob[-1][:2][::-1], center_2, atol=atol
        ):
            errors += 1

    # we do not expect more than 5% errors
    # (there are 100 frames and 2 blobs per frame)
    assert errors < 10, (
        f"More than 5% errors ({errors}) in derotation "
        + f" with wrong center of rotation {center_of_rotation_initial}"
    )
