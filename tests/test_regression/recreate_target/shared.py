from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.simulate.basic_rotator import Rotator

NUMBER_OF_FRAMES = 3


def square_with_gray_stripes_in_black_background() -> np.ndarray:
    """
    Create a square with gray stripes in a black background.
    It's the same stripe pattern of the example script
    `rotate_and_derotate_a_square`. This is going to be the image
    that will be rotated at different angles.

    Returns
    -------
    image : np.ndarray
    """

    # initialize image
    image = np.zeros((100, 100))

    # create gray stripes
    gray_values = [i % 5 * 60 + 15 for i in range(100)]
    for i in range(100):
        image[i] = gray_values[i]

    # create a square by setting borders to zero
    image[:20] = 0
    image[-20:] = 0
    image[:, :20] = 0
    image[:, -20:] = 0

    return image


def get_image_stack_and_angles() -> Tuple[np.ndarray, np.ndarray]:
    """Take the image of a square with gray stripes in a black background
    and create a stack of 3 images with the same content. This is going to
    emulate a "video" of 3 frames acquired without any rotation.

    Also, create an array of angles that will be used to rotate the images.
    In this case, the angles are going to be integres from 0 to 299. Each
    rotation angle will be applied to a line of the image. Therefore, the
    total number of angles is going to be the number of lines in the image.

    These two arrays are going to be used to test the Rotator class to create
    the rotated "video".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple with a static image stack (n_frames x n_lines x
        n_pixels_in_line) of three frames and the angles (n_frames * n_lines).
    """
    # get the image
    image = square_with_gray_stripes_in_black_background()

    # create a stack of 3 images
    stack = np.array([image for _ in range(NUMBER_OF_FRAMES)])

    # create an array of increasing angles
    n_total_lines = stack.shape[0] * stack.shape[1]
    angles = np.arange(n_total_lines)

    return stack, angles


def rotate_images(
    image_stack: np.ndarray,
    angles: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Rotate a stack of images by a given set of angles.
    This function emulates the acquisition of a "video" of images
    in a line scannning microscope.

    Parameters
    ----------
    image_stack : np.ndarray
        The stack of images to be rotated. The shape of the stack
        should be (n_frames x n_lines x n_pixels_in_line).
    angles : np.ndarray
        The angles to be used to rotate the images. The length of the
        array should be the number of lines in the image
    center : Tuple[int, int], optional
        The center of rotation. If None, the center is going to be
        the center of the image, by default None

    Returns
    -------
    np.ndarray
        The rotated image stack.
    """

    # initialize the Rotator object and rotate the images
    rotator = Rotator(angles, image_stack, center=center)
    return rotator.rotate_by_line()


def load_rotated_images(
    directory: str, len_stack: int, center: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Load pre-computed rotated images from a directory.
    These images are going to be used to compare current
    outputs of rotation and derotation functions.

    Parameters
    ----------
    directory : str
        The path where the images are stored.
    len_stack : int
        The number of images in the stack (i.e., the number of frames).
    center : Tuple[int, int], optional
        The center of rotation. If None, the center is going to be
        the center of the image, by default None

    Returns
    -------
    np.ndarray
        The stack of rotated images.
    """
    #  expected end of file name
    center_suffix = "" if center is None else f"{center[0]}_{center[1]}_"

    #  load pre-computed images one by one
    rotated_image_stack = []
    for i in range(1, len_stack + 1):
        image_path = Path(directory) / f"rotated_frame_{center_suffix}{i}.png"
        rotated_image = Image.open(image_path).convert("L")
        rotated_image_stack.append(np.array(rotated_image))

    #  convert to numpy array and return
    return np.array(rotated_image_stack)


def regenerate_rotator_images_for_testing(
    image_stack: np.ndarray,
    angles: np.ndarray,
    center: Optional[Tuple[int, int]] = None,
):
    """Regenerate rotated images for a given center.

    Parameters
    ----------
    image_stack : np.ndarray
        The stack of images to be rotated. The shape of the stack
        should be (n_frames x n_lines x n_pixels_in_line).
    angles : np.ndarray
        The angles to be used to rotate the images. The length of the
        array should be the number of lines in the image
    center : Tuple[int, int], optional
        The center of rotation. If None, the center is going to be
        the center of the image, by default None
    """

    # Rotate the images using the Rotator
    rotated_image_stack = rotate_images(image_stack, angles, center=center)

    # expected end of file name
    center_suffix = "" if center is None else f"{center[0]}_{center[1]}_"

    # create a directory to store the images if it does not existå
    path = Path("tests/test_regression/images/rotator")
    path.mkdir(parents=True, exist_ok=True)

    # Save rotated images
    for i, rotated_frame in enumerate(rotated_image_stack):
        rotated_image = Image.fromarray(rotated_frame.astype("uint8"))
        rotated_image.save(path / f"rotated_frame_{center_suffix}{i + 1}.png")


def regenerate_derotated_images_for_testing(
    rotated_image_stack: np.ndarray,
    angles: np.ndarray,
    output_directory: str,
    center: Optional[Tuple[int, int]] = None,
):
    """Derotate a stack of images by a given set of angles.

    Parameters
    ----------
    rotated_image_stack : np.ndarray
        The stack of images to be derotated. The shape of the stack
        should be (n_frames x n_lines x n_pixels_in_line).
    angles : np.ndarray
        The angles to be used to derotate the images.
    output_directory : str
        The path where the derotated images are going to be saved.
    center : Tuple[int, int], optional
        The center of rotation. If None, the center is going to be
        the center of the image, by default None
    """
    # Derotate by line the rotated images
    derotated_image_stack = derotate_an_image_array_line_by_line(
        rotated_image_stack, angles, center=center
    )

    # Save derotated images
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    center_suffix = "" if center is None else f"{center[0]}_{center[1]}_"

    for i, derotated_frame in enumerate(derotated_image_stack):
        derotated_image = Image.fromarray(derotated_frame.astype("uint8"))
        derotated_image.save(
            Path(output_directory)
            / f"derotated_frame_{center_suffix}{i + 1}.png"
        )
