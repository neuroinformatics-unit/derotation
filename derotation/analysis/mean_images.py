"""
This module contains functions to calculate  mean images
in steps of 10 degrees. It is used when finding the center of rotation.
"""

import copy
import logging

import numpy as np


def calculate_mean_images(
    image_stack: np.ndarray, rot_deg_frame: np.ndarray, round_decimals: int = 2
) -> np.ndarray:
    """Calculate the mean images for each rotation angle. This required
    to calculate the shifts using phase cross correlation.

    Parameters
    ----------
    rotated_image_stack : np.ndarray
        The rotated image stack.
    rot_deg_frame : np.ndarray
        The rotation angles for each frame.
    round_decimals : int, optional
        The number of decimals to round the angles to, by default 2

    Returns
    -------
    np.ndarray
        The mean images for each rotation angle.
    """
    #  correct for a mismatch in the total number of frames
    angles_subset = copy.deepcopy(rot_deg_frame)
    if len(angles_subset) > len(image_stack):
        angles_subset = angles_subset[: len(image_stack)]
    else:
        image_stack = image_stack[: len(angles_subset)]

    assert len(image_stack) == len(angles_subset), (
        "Mismatch in the number of images and angles"
    )

    rounded_angles = np.round(angles_subset, round_decimals)

    mean_images = []
    for i in np.arange(10, 360, 10):
        images = image_stack[rounded_angles == i]
        if len(images) == 0:
            # np.mean of no images is an all-NaN image, not an error
            logging.warning(f"Angle {i} not found in the image stack")
            example_angles = np.random.choice(rounded_angles, 100)
            logging.info(f"Example angles: {example_angles}")
            continue

        mean_images.append(np.mean(images, axis=0))

    return np.asarray(mean_images)
