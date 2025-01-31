#  -----------------------------------------------------
#  This test is an integration test that combines the incremental derotation
#  pipeline with the full derotation pipeline. The test generates a test image
#  stack with two circles and rotates it incrementally and sinusoidally using
#  a variable center of rotation.
#
#  The image looks like this:
#
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ███████████████████▒▒▓██████████████████
#     ██████████████████▒  ░██████████████████
#     ██████████████████▓░░▒██████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#     ███████████████████████████████▓▓███████
#     ██████████████████████████████▓▒▒▓██████
#     ███████████████████████████████▓▓▓██████
#     ████████████████████████████████████████
#     ████████████████████████████████████████
#
#  The center of rotation is then estimated by the incremental pipeline and
#  used to derotate the sinusoidal stack. The test checks if the two circles
#  are detected in the derotated stack. The test is parametrized with two
#  different initial centers of rotation.
#
#  This test can also be run as a standalone script to generate debugging
#  plots.
#
#  Sections:
#  1. Prepare the 3D image stack and the rotation angles
#  2. Integration pipeline with mock of the IncrementalPipeline
#  3. Debugging plots
#  4. Test the integration pipeline
#  5. Run the integration pipeline as a standalone script
#  -----------------------------------------------------


import copy
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from skimage.feature import blob_log

from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)
from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.simulate.line_scanning_microscope import Rotator

#  -----------------------------------------------------
#  Prepare the 3D image stack and the rotation angles
#  -----------------------------------------------------


def create_sample_image_with_two_cells(
    center_of_bright_cell: Tuple[int, int] = (50, 10),
    center_of_dimmer_cell: Tuple[int, int] = (60, 60),
    lines_per_frame: int = 100,
    second_cell=True,
    radius: int = 5,
) -> np.ndarray:
    """Create a 2D image with two circles, one bright and one dim (optional)
    by default in the top center and bottom right, respectively.

    Location of the circles can be changed by providing the
    center_of_bright_cell and center_of_dimmer_cell parameters.

    Parameters
    ----------
    center_of_bright_cell : Tuple[int, int], optional
        Location of brightest cell, by default (50, 10)
    center_of_dimmer_cell : Tuple[int, int], optional
        Location of dimmer cell, by default (60, 60)
    lines_per_frame : int, optional
        Number of lines per frame, by default 100
    second_cell : bool, optional
        Add an extra dimmer cell, by default True
    radius : int, optional
        Radius of the circles, by default 5

    Returns
    -------
    np.ndarray
        2D image with two circles, one bright and one dim
    """

    # Initialize a black image of size 100x100
    image = np.zeros((lines_per_frame, lines_per_frame), dtype=np.uint8)

    # Define the circle's parameters
    white_value = 255  # white color for the circle

    # Draw a white circle in the top center
    y, x = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (x - center_of_bright_cell[0]) ** 2 + (
        y - center_of_bright_cell[1]
    ) ** 2 <= radius**2
    image[mask] = white_value

    if second_cell:
        #  add an extra gray circle at the bottom right
        gray_value = 128
        # Draw a gray circle in the bottom right
        mask2 = (x - center_of_dimmer_cell[0]) ** 2 + (
            y - center_of_dimmer_cell[1]
        ) ** 2 <= radius**2
        image[mask2] = gray_value

    return image


def create_image_stack(image: np.ndarray, num_frames: int) -> np.ndarray:
    """Create a 3D image stack by repeating the 2D image
    for a given number of frames.

    Parameters
    ----------
    image : np.ndarray
        A 2D image
    num_frames : int
        Number of frames in the 3D image stack

    Returns
    -------
    np.ndarray
        3D image stack
    """
    return np.array([image for _ in range(num_frames)])


def create_rotation_angles(
    image_stack_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create rotation angles for incremental and sinusoidal rotation
    for a given 3D image stack.

    Parameters
    ----------
    image_stack_shape : Tuple[int, int]
        Shape of the 3D image stack

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of incremental and sinusoidal rotation angles
    """

    # Generate rotation angles for incremental rotation
    # which consists of 36 steps of 10 degrees each
    # If plotted they look like a staircase
    num_lines_total = image_stack_shape[1] * image_stack_shape[0]
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

    # Generate rotation angles for sinusoidal rotation
    max_rotation = 360  # max rotation angle
    num_cycles = 1
    sinusoidal_angles = max_rotation * np.sin(
        np.linspace(0, num_cycles * 2 * np.pi, num_lines_total)
    )

    return incremental_angles.astype("float64"), sinusoidal_angles


#  -----------------------------------------------------
#  Integration pipeline with mock of the IncrementalPipeline
#  -----------------------------------------------------


def get_center_of_rotation(
    rotated_stack_incremental: np.ndarray, incremental_angles: np.ndarray
) -> Tuple[int, int]:
    """Get the center of rotation by using the IncrementalPipeline.

    The Incremental pipeline has the responsibility to find the center of
    rotation but with mock data we cannot use it off the shelf because it
    is too bound to signals coming from a real motor in the
    `calculate_mean_images` method and in the constructor.
    We will create a mock class that inherits from the IncrementalPipeline
    and overwrite the `calculate_mean_images` method to work with our mock
    data.

    Parameters
    ----------
    rotated_stack_incremental : np.ndarray
        The 3D image stack rotated incrementally
    incremental_angles : np.ndarray
        The rotation angles for incremental rotation

    Returns
    -------
    Tuple[int, int]
        The center of rotation
    """

    # Mock class to use the IncrementalPipeline
    class MockIncrementalPipeline(IncrementalPipeline):
        def __init__(self):
            # Overwrite the constructor and provide the mock data
            self.image_stack = rotated_stack_incremental
            self.rot_deg_frame = incremental_angles[
                :: rotated_stack_incremental.shape[1]
            ]
            self.num_frames = rotated_stack_incremental.shape[0]

            if __name__ == "__main__":
                self.debugging_plots = True
                self.debug_plots_folder = Path("debug/")
            else:
                self.debugging_plots = False

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

    # Use the mock class to find the center of rotation
    pipeline = MockIncrementalPipeline()
    center_of_rotation = pipeline.find_center_of_rotation()

    return center_of_rotation


def integration_pipeline(
    test_image: np.ndarray,
    center_of_rotation_initial: Tuple[int, int],
    num_frames: int,
) -> np.ndarray:
    """Integration pipeline that combines the incremental and sinusoidal
    rotation pipelines to derotate a 3D image stack.

    The pipeline rotates the image stack incrementally and sinusoidally
    and then derotates the sinusoidal stack using the center of rotation
    estimated by the incremental pipeline.

    Parameters
    ----------
    test_image : np.ndarray
        A 2D image
    center_of_rotation_initial : Tuple[int, int]
        Initial center of rotation
    num_frames : int
        Number of frames in the 3D image stack

    Returns
    -------
    np.ndarray
        Derotated 3D image stack
    """

    # -----------------------------------------------------
    # Create the 3D image stack
    image_stack = create_image_stack(test_image, num_frames)

    # Generate rotation angles
    incremental_angles, sinusoidal_angles = create_rotation_angles(
        image_stack.shape
    )

    #  -----------------------------------------------------
    # Use the Rotator to create the rotated image stacks

    # Initialize Rotator for incremental rotation
    rotator_incremental = Rotator(
        incremental_angles, image_stack, center_of_rotation_initial
    )
    # Rotate the image stack incrementally
    rotated_stack_incremental = rotator_incremental.rotate_by_line()

    # Initialize Rotator for sinusoidal rotation
    rotator_sinusoidal = Rotator(
        sinusoidal_angles, image_stack, center_of_rotation_initial
    )
    # Rotate the image stack sinusoidally
    rotated_stack_sinusoidal = rotator_sinusoidal.rotate_by_line()

    #  -----------------------------------------------------
    # Derotate the sinusoidal stack using the center of rotation
    # estimated by the incremental pipeline

    # Get the center of rotation with a mock of the IncrementalPipeline
    center_of_rotation = get_center_of_rotation(
        rotated_stack_incremental, incremental_angles
    )

    # Derotate the sinusoidal stack
    derotated_sinusoidal = derotate_an_image_array_line_by_line(
        rotated_stack_sinusoidal,
        sinusoidal_angles,
        center=center_of_rotation,
    )

    #  -----------------------------------------------------
    #  Debugging plots
    #  Will be run if the script is run as a standalone script
    if __name__ == "__main__":
        plot_angles(incremental_angles, sinusoidal_angles)
        plot_a_few_rotated_frames(
            rotated_stack_incremental, rotated_stack_sinusoidal
        )
        plot_derotated_frames(derotated_sinusoidal)

    return derotated_sinusoidal


# -----------------------------------------------------
# Debugging plots
# -----------------------------------------------------


def plot_angles(incremental_angles: np.ndarray, sinusoidal_angles: np.ndarray):
    """Plot the incremental and sinusoidal rotation angles.

    Parameters
    ----------
    incremental_angles : np.ndarray
        Incremental rotation angles
    sinusoidal_angles : np.ndarray
        Sinusoidal rotation angles
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle("Rotation Angles")

    axs[0].plot(incremental_angles, label="Incremental Rotation")
    axs[0].set_title("Incremental Rotation Angles")
    axs[0].set_ylabel("Angle (degrees)")
    axs[0].set_xlabel("Line Number")
    axs[0].legend()

    axs[1].plot(sinusoidal_angles, label="Sinusoidal Rotation")
    axs[1].set_title("Sinusoidal Rotation Angles")
    axs[1].set_ylabel("Angle (degrees)")
    axs[1].set_xlabel("Line Number")
    axs[1].legend()

    plt.tight_layout()

    plt.savefig("debug/rotation_angles.png")

    plt.close()


def plot_a_few_rotated_frames(
    rotated_stack_incremental: np.ndarray,
    rotated_stack_sinusoidal: np.ndarray,
):
    """Plot a few frames from the rotated stacks.

    Parameters
    ----------
    rotated_stack_incremental : np.ndarray
        The 3D image stack rotated incrementally
    rotated_stack_sinusoidal : np.ndarray
        The 3D image stack rotated sinusoidally
    """

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axs[0]):
        ax.imshow(rotated_stack_incremental[i * 5], cmap="gray")
        ax.set_title(f"Frame {i * 5}")
        ax.axis("off")

    for i, ax in enumerate(axs[1]):
        ax.imshow(rotated_stack_sinusoidal[i * 5], cmap="gray")
        ax.set_title(f"Frame {i * 5}")
        ax.axis("off")

    plt.savefig("debug/rotated_stacks.png")

    plt.close()


def plot_derotated_frames(derotated_sinusoidal: np.ndarray):
    """Plot a few frames from the derotated stack.

    Parameters
    ----------
    derotated_sinusoidal : np.ndarray
        The 3D image stack derotated sinusoidally
    """

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axs[0]):
        ax.imshow(derotated_sinusoidal[i * 5], cmap="gray")
        ax.set_title(f"Frame {i * 5}")
        ax.axis("off")

    for i, ax in enumerate(axs[1]):
        ax.imshow(derotated_sinusoidal[i * 5 + 1], cmap="gray")
        ax.set_title(f"Frame {i * 5 + 1}")
        ax.axis("off")

    plt.savefig("debug/derotated_sinusoidal.png")

    plt.close()


# -----------------------------------------------------
# Test the integration pipeline
# -----------------------------------------------------


@pytest.mark.parametrize("center_of_rotation_initial", [(44, 51), (51, 44)])
def test_blob_detection_on_derotated_stack(
    center_of_rotation_initial: Tuple[int, int],
):
    """Test if the two circles are detected in the derotated stack
    at the expected locations when the center of rotation is estimated
    with the incremental pipeline. It is parametrized with two different
    initial centers of rotation.

    It allows some error tolerance in the center of the detected blobs,
    precisely 6 pixels in any direction. This is because the derotation
    itself would loose some information and the reconstructed blobs might
    miss pixels and consequently be detected at slightly different locations.
    Test will fail with more than 5% errors.

    Parameters
    ----------
    center_of_rotation_initial : Tuple[int, int]
        Center of rotation
    """

    # -----------------------------------------------------
    # Setting up the test

    # Set the centers of the two circles and the number of frames
    center_1 = (50, 10)
    center_2 = (60, 60)
    num_frames = 100

    # Create a test image with two circles
    test_image = create_sample_image_with_two_cells(
        center_of_bright_cell=center_1, center_of_dimmer_cell=center_2
    )

    # Run the integration pipeline and obtain the derotated stack
    derotated_sinusoidal = integration_pipeline(
        test_image, center_of_rotation_initial, num_frames
    )

    # -----------------------------------------------------
    # Are the blobs detected in the derotated stack where we expect them?
    # If yes, the derotation was successful and the test passes
    # If not, the derotation was not successful and the test fails

    # Detect the blobs in the derotated stack
    blobs = [
        blob_log(img, min_sigma=3, max_sigma=5) for img in derotated_sinusoidal
    ]

    # Get the center of the blobs
    # for every frame, place first the blob with the smallest x value
    blobs = [sorted(blob, key=lambda x: x[1]) for blob in blobs]

    # Compare the first and second blob to the expected values
    errors = 0
    atol = 6  # 6 pixels tolerance for the center of the blobs
    for blob in blobs:
        if not np.allclose(blob[0][:2][::-1], center_1, atol=atol):
            errors += 1
        if len(blob) > 1 and not np.allclose(
            blob[-1][:2][::-1], center_2, atol=atol, rtol=0
        ):
            errors += 1

    # we do not expect more than 5% errors
    # (there are 100 frames and 2 blobs per frame)
    assert errors < derotated_sinusoidal.shape[0] * 2 * 0.05, (
        f"More than 5% errors ({errors}) in derotation "
        + f" with wrong center of rotation {center_of_rotation_initial}"
    )


# -----------------------------------------------------
# Run the integration pipeline as a standalone script
# to generate debugging plots
# -----------------------------------------------------

if __name__ == "__main__":
    Path("debug/").mkdir(parents=True, exist_ok=True)
    test_blob_detection_on_derotated_stack((44, 51))
