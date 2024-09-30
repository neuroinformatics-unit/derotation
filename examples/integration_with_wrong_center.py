import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)
from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.simulate.basic_rotator import Rotator


def create_test_image():
    # Initialize a black image of size 100x100
    image = np.zeros((100, 100), dtype=np.uint8)

    # Define the circle's parameters
    center = (50, 10)  # (x, y) position
    radius = 5  # radius of the circle
    white_value = 255  # white color for the circle

    #  add an extra gray circle at the bottom right
    center2 = (80, 80)
    radius2 = 5
    gray_value = 128

    # Draw a white circle in the top center
    y, x = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
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

    # if len incremental angles is not equal to num_lines_total, repeat the last angle
    if len(incremental_angles) < num_lines_total:
        incremental_angles = np.concatenate(
            (
                incremental_angles,
                [incremental_angles[-1]]
                * (num_lines_total - len(incremental_angles)),
            )
        )

    max_rotation = 360  # max rotation angle
    num_cycles = 7
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
            #  Overwrite original method as it is too bound to signal coming from a real motor
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


def main():
    # Create the test image
    test_image = create_test_image()
    # save the test image
    plt.imsave("debug/test_image.png", test_image, cmap="gray")

    # Create a stack of identical frames
    num_frames = 1000
    image_stack = create_image_stack(test_image, num_frames)

    # Generate rotation angles
    incremental_angles, sinusoidal_angles = create_rotation_angles(image_stack)

    # plot the angles
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
    #  save plot
    plt.savefig("debug/rotation_angles.png")

    plt.close()

    # Use the Rotator to create the rotated image stacks
    rotator_incremental = Rotator(incremental_angles, image_stack)
    rotated_stack_incremental = rotator_incremental.rotate_by_line()

    rotator_sinusoidal = Rotator(sinusoidal_angles, image_stack)
    rotated_stack_sinusoidal = rotator_sinusoidal.rotate_by_line()

    #  save the two stacka as tiffs
    tiff.imwrite(
        "debug/rotated_stack_incremental.tiff", rotated_stack_incremental
    )
    tiff.imwrite(
        "debug/rotated_stack_sinusoidal.tiff", rotated_stack_sinusoidal
    )

    center_of_rotation = get_center_of_rotation(
        rotated_stack_incremental, incremental_angles
    )

    derotated_sinusoidal = derotate_an_image_array_line_by_line(
        rotated_stack_sinusoidal,
        sinusoidal_angles,
        center=center_of_rotation,
    )

    #  save the derotated stack as tiff
    tiff.imwrite("debug/derotated_sinusoidal.tiff", derotated_sinusoidal)

    # Optional: Visualize synthetic stacks for verification in a few frames
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    #  show incremental and sinusoidal rotation

    for i, ax in enumerate(axs[0]):
        ax.imshow(rotated_stack_incremental[i * 50], cmap="gray")
        ax.set_title(f"Frame {i * 24}")
        ax.axis("off")

    for i, ax in enumerate(axs[1]):
        ax.imshow(rotated_stack_sinusoidal[i * 50], cmap="gray")
        ax.set_title(f"Frame {i * 24}")
        ax.axis("off")

    # Save the plot
    plt.savefig("debug/rotated_stacks.png")

    plt.close()

    # Let's look at 10 frames of the derotated sinusoidal stack
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axs[0]):
        ax.imshow(derotated_sinusoidal[i * 50], cmap="gray")
        ax.set_title(f"Frame {i * 24}")
        ax.axis("off")

    for i, ax in enumerate(axs[1]):
        ax.imshow(derotated_sinusoidal[i * 50 + 1], cmap="gray")
        ax.set_title(f"Frame {i * 24 + 1}")
        ax.axis("off")

    # Save the plot
    plt.savefig("debug/derotated_sinusoidal.png")

    plt.close()


if __name__ == "__main__":
    main()
