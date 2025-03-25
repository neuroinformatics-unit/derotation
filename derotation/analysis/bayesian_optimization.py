"""
Bayesian optimization to find the best center of rotation.
It optimizes the center of rotation by running first the entire derotation
process and then calculating the point to point distance of the most detected
blob in the mean images, calculated at angle steps of 10 degrees.

Steps in the optimization:
    - Derotate the movie.
    - Calculate the mean images.
    - Calculate the point to point distance of the most detected blob.
    - Return the negative of the point to point distance as the metric to
      maximize.

The call to ``ptd_of_most_detected_blob`` can output debugging plots.

The optimization is done using the BayesianOptimization class from the
``bayes_opt`` library. The results of the optimization are printed to the
console.
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization

from derotation.analysis.mean_images import calculate_mean_images
from derotation.analysis.metrics import ptd_of_most_detected_blob
from derotation.derotate_by_line import derotate_an_image_array_line_by_line


class BO_for_derotation:
    """
    Bayesian optimization to find the best center of rotation.

    Parameters
    ----------
    movie : np.ndarray
        The calcium imaging movie to derotate.
    rot_deg_line : np.ndarray
        The rotation angles of the rotation plane.
    rot_deg_frame : np.ndarray
        The rotation angles of the frames.
    blank_pixels_value : float
        The value of the blank pixels.
    center : Tuple[int, int]
        The initial center of rotation.
    delta : int
        The variation allowed in the boundaries of the center of rotation.
    init_points : int, optional
        The number of initial points to evaluate, by default 2
    n_iter : int, optional
        The number of iterations to run the optimization, by default 10
    debug_plots_folder : Path, optional
        The folder to save the debugging plots, by default
        Path("./debug_plots")
    """

    def __init__(
        self,
        movie: np.ndarray,
        rot_deg_line: np.ndarray,
        rot_deg_frame: np.ndarray,
        blank_pixels_value: float,
        center: Tuple[int, int],
        delta: int,
        init_points: int = 2,
        n_iter: int = 10,
        debug_plots_folder: Path = Path("./debug_plots"),
    ):
        """Initializes the BO_for_derotation class."""
        self.movie = movie
        self.rot_deg_line = rot_deg_line
        self.rot_deg_frame = rot_deg_frame
        self.blank_pixels_value = blank_pixels_value
        x, y = center
        self.pbounds = {
            "x": (x - delta, x + delta),
            "y": (y - delta, y + delta),
        }
        self.init_points = init_points
        self.n_iter = n_iter
        self.debug_plots_folder = debug_plots_folder

        self.subfolder = self.debug_plots_folder / "bo"
        Path(self.subfolder).mkdir(parents=True, exist_ok=True)

    def optimize(self):
        """Optimize the center of rotation."""

        def derotate_and_get_metric(
            x: float,
            y: float,
        ):
            derotated_chunk = derotate_an_image_array_line_by_line(
                image_stack=self.movie,
                rot_deg_line=self.rot_deg_line,
                blank_pixels_value=self.blank_pixels_value,
                center=(int(x), int(y)),
            )

            mean_images = calculate_mean_images(
                derotated_chunk, self.rot_deg_frame, round_decimals=0
            )

            plt.imshow(mean_images[0], cmap="gray")
            plt.savefig(self.subfolder / f"mean_image_0_{x:.2f}_{y:.2f}.png")
            plt.close()

            ptd = ptd_of_most_detected_blob(
                mean_images,
                debug_plots_folder=self.subfolder,
                image_names=[
                    f"blobs_{x:.2f}_{y:.2f}.png",
                    f"blob_centers_{x:.2f}_{y:.2f}.png",
                ],
            )

            # we are maximizing the metric, so
            # we need to return the negative of the metric
            return -ptd

        optimizer = BayesianOptimization(
            f=derotate_and_get_metric,
            pbounds=self.pbounds,
            verbose=2,
            random_state=1,
        )

        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )

        for i, res in enumerate(optimizer.res):
            logging.info(f"Iteration {i}: {res}")

        return optimizer.max
