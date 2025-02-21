import numpy as np
from bayes_opt import BayesianOptimization

from derotation.analysis.mean_images import calculate_mean_images
from derotation.analysis.metrics import ptd_of_most_detected_blob
from derotation.derotate_by_line import derotate_an_image_array_line_by_line
import logging

class BO_for_derotation:
    def __init__(
        self,
        movie_chunck: np.ndarray,
        angle_array_chunk: np.ndarray,
        blank_pixels_value: float,
        center: int,
        delta: int,
        init_points: int = 2,
        n_iter: int = 10,
    ):
        self.movie_chunck = movie_chunck
        self.angle_array_chunk = angle_array_chunk
        self.blank_pixels_value = blank_pixels_value
        x, y = center
        self.pbounds = {
            "x": (x - delta, x + delta),
            "y": (y - delta, y + delta),
            "theta": (0, 360),  # plane angle in degrees
            "phi": (0, 360),  # ellipse rotation angle in degrees
        }
        self.init_points = init_points
        self.n_iter = n_iter

    def optimize(self):
        def derotate_and_get_metric(
            x: float,
            y: float,
            theta: float,
            phi: float,
        ):
            derotated_chunk = derotate_an_image_array_line_by_line(
                image_stack=self.movie_chunck,
                rot_deg_line=self.angle_array_chunk,
                blank_pixels_value=self.blank_pixels_value,
                center=(x, y),
                use_homography=True,
                rotation_plane_angle=theta,
                rotation_plane_orientation=phi,
            )

            mean_images = calculate_mean_images(derotated_chunk, self.angle_array_chunk)

            
            ptd = ptd_of_most_detected_blob(mean_images, plot=False)

            # we are unfortunately maximizing the metric, so
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
