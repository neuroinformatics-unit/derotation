"""
This module contains the BlobDetection class, which is used to detect the
largest blob in each image of an image stack.
The class uses the ``blob_log`` function from the
skimage.feature module to detect the blobs. The coordinates of the largest
blob in each image are returned as a numpy array. The class also has a method
to plot the first 4 blobs in each image, which is useful for debugging
purposes.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_log
from tqdm import tqdm


class BlobDetection:
    """
    The BlobDetection class is used to detect the largest blob in each image
    of an image stack.

    Parameters
    ----------
    debugging_plots : bool, optional
        Whether to create debugging plots, by default False
    debug_plots_folder : Path, optional
        The folder to save the debugging plots, by default None
    blob_log_params : dict, optional
        The parameters for the blob detection, by default
        {"max_sigma": 12, "min_sigma": 7, "threshold": 0.95, "overlap": 0}
        which are the parameters that worked best for the 3-photon data.
    """

    def __init__(
        self,
        debugging_plots: bool = False,
        debug_plots_folder: Path = Path("debug"),
        blob_log_params: dict = {
            "max_sigma": 12,
            "min_sigma": 7,
            "threshold": 0.95,
            "overlap": 0,
        },
    ):
        """Initializes the BlobDetection class."""
        self.debugging_plots = debugging_plots
        self.debug_plots_folder = debug_plots_folder
        self.blob_log_params = blob_log_params

    def get_coords_of_largest_blob(
        self, image_stack: np.ndarray
    ) -> np.ndarray:
        """Get the coordinates of the largest blob in each image.

        Parameters
        ----------
        image_stack : np.ndarray
            The image stack.

        Returns
        -------
        np.ndarray
            The coordinates of the largest blob in each image.
        """

        blobs = [
            blob_log(img, **self.blob_log_params) for img in tqdm(image_stack)
        ]

        # sort blobs by size
        blobs = [
            blobs[i][blobs[i][:, 2].argsort()] for i in range(len(image_stack))
        ]

        coord_first_blob_of_every_image = []
        for i, blob in enumerate(blobs):
            if len(blob) > 0:
                coord_first_blob_of_every_image.append(blob[0][:2].astype(int))
            else:
                coord_first_blob_of_every_image.append([np.nan, np.nan])
                logging.warning(f"No blobs were found in image {i}")

        #  invert x, y order
        coord_first_blob_of_every_image = [
            (coord[1], coord[0]) for coord in coord_first_blob_of_every_image
        ]

        # plot blobs on top of every frame
        if self.debugging_plots:
            self.plot_blob_detection(blobs, image_stack)

        return np.asarray(coord_first_blob_of_every_image)

    def plot_blob_detection(self, blobs: list, image_stack: np.ndarray):
        """Plot the first 4 blobs in each image. This is useful to check if
        the blob detection is working correctly and to see if the identity of
        the largest blob is consistent across the images.

        Parameters
        ----------
        blobs : list
            The list of blobs in each image.
        image_stack : np.ndarray
            The image stack.
        """

        fig, ax = plt.subplots(4, 3, figsize=(10, 10))
        for i, a in tqdm(enumerate(ax.flatten())):
            a.imshow(image_stack[i])
            a.set_title(f"{i * 5} degrees")
            a.axis("off")

            for j, blob in enumerate(blobs[i][:4]):
                y, x, r = blob
                c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
                a.add_patch(c)
                a.text(x, y, str(j), color="red")

        # save the plot
        Path(self.debug_plots_folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(self.debug_plots_folder / "blobs.png")
        plt.close()
