import copy
import logging
import numpy as np


def calculate_mean_images(
    image_stack: np.ndarray, 
    rot_deg_frame: np.ndarray,
    round_decimals: int = 2
) -> np.ndarray:
    """Calculate the mean images for each rotation angle. This required
    to calculate the shifts using phase cross correlation.

    Parameters
    ----------
    rotated_image_stack : np.ndarray
        The rotated image stack.

    Returns
    -------
    np.ndarray
        The mean images for each rotation angle.
    """
    logging.info("Calculating mean images...")

    #  correct for a mismatch in the total number of frames
    angles_subset = copy.deepcopy(rot_deg_frame)
    angles_subset = angles_subset[: len(image_stack)]

    # also there is a bias on the angles
    angles_subset += -0.1
    rounded_angles = np.round(angles_subset, round_decimals)

    mean_images = []
    for i in np.arange(10, 360, 10):
        try:
            images = image_stack[rounded_angles == i]
            mean_image = np.mean(images, axis=0)

            mean_images.append(mean_image)
        except IndexError:
            logging.warning(f"Angle {i} not found in the image stack")

    return np.asarray(mean_images)
