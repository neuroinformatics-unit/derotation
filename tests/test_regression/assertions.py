from pathlib import Path

import numpy as np
from PIL import Image


def comapre_images(
    image_idx: int,
    current_image: np.ndarray,
    target_image: np.ndarray,
    atol: float,
    save_location: Path,
    wrong_pixel_tollerance: int = 0,
):
    """Handle the comparison of two images and save the image if they are
    different. It uses the np.allclose function to compare the two images
    with a given tolerance level. If the images are not close, it calculates
    the difference and saves the wrong image for further inspection. If the
    number of wrong pixels is greater than the tollerance level, it raises
    an assertion error.

    Parameters
    ----------
    image_idx : int
        The index of the image in the image stack - it will be the iteration
        number.
    current_image : np.ndarray
        The image that is currently being compared.
    target_image : np.ndarray
        The target image that the current image is compared against.
    atol : float
        The tolerance level for the comparison.
    save_location : Path
        The location where the wrong image will be saved.
    wrong_pixel_tollerance : int, optional
        The number of wrong pixels that are allowed, by default 0

    Raises
    ------
    AssertionError
        If the current image is not close to the target image within the
        tolerance level.
    """

    try:
        # Check if the current image is close to the target image
        # within the tolerance level
        assert np.allclose(current_image, target_image, atol=atol)

    except AssertionError:
        # If the current image is not close to the target image
        # calculate the difference and save the image for further inspection
        diff = np.abs(current_image - target_image)
        indexes = np.where(diff > 1)
        rotated_image = Image.fromarray(current_image.astype("uint8"))
        rotated_image.save(
            save_location / f"rotated_frame_{image_idx + 1}_wrong.png"
        )

        #  If we do not expect any wrong pixels, raise an assertion error
        if indexes.shape[0] > wrong_pixel_tollerance:
            assert False, (
                "Regenerated image is different from the target image. "
                + "Index where it is different: "
                + f"{indexes}, Total: {len(indexes[0])}"
            )
