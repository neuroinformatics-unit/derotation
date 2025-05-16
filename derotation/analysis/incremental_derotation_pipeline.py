"""
This module contains the ``IncrementalPipeline`` class, which is a child of
the ``FullPipeline`` class. It is used to derotate the image stack that was
acquired using the incremental rotation method. The class inherits all the
attributes and methods from the ``FullPipeline`` class and adds additional
logic to register the images.
Methods that are not overwritten from the parent class are not documented
here.
"""

import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from derotation.analysis.blob_detection import BlobDetection
from derotation.analysis.fit_ellipse import (
    fit_ellipse_to_points,
    plot_ellipse_fit_and_centers,
)
from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.analysis.mean_images import calculate_mean_images


class IncrementalPipeline(FullPipeline):
    """Derotate the image stack that was acquired using the incremental
    rotation method.

    As a child of ``FullPipeline``, it inherits all the attributes and
    methods from it and adds additional logic to register the images.
    Here in the constructor we specify the degrees for each incremental
    rotation and the number of rotations.
    """

    ### ------- Methods to overwrite from the parent class ------------ ###
    def __init__(self, *args, **kwargs):
        """Initialize the IncrementalPipeline object."""
        super().__init__(*args, **kwargs)

        self.degrees_per_small_rotation = 10
        self.number_of_rotations = (
            self.rot_deg // self.degrees_per_small_rotation
        )

    def __call__(self):
        """Overwrite the ``__call__`` method from the parent class to derotate
        the image stack acquired using the incremental rotation method.
        After processing the analog signals, the image stack is rotated by
        frame and then registered using phase cross correlation.
        """
        self.process_analog_signals()
        self.offset = self.find_image_offset(self.image_stack[0])

        rotated_images = self.derotate_frames_line_by_line()
        self.masked_image_volume = self.add_circle_mask(
            rotated_images, self.mask_diameter
        )

        self.save(self.masked_image_volume)
        self.save_csv_with_derotation_data()

    def create_signed_rotation_array(self) -> np.ndarray:
        logging.info("Creating signed rotation array...")
        rotation_on = np.zeros(self.total_clock_time)
        for i, (start, end) in enumerate(
            zip(
                self.rot_blocks_idx["start"],
                self.rot_blocks_idx["end"],
            )
        ):
            rotation_on[start:end] = np.ones(end - start) * -1  # -1

        return rotation_on

    def is_number_of_ticks_correct(self) -> bool:
        """Overwrite the method from the parent class to check if the number
        of ticks is as expected.

        Returns
        -------
        bool
            True if the number of ticks is as expected, False otherwise.
        """
        self.expected_tiks_per_rotation = (
            self.rot_deg / self.rotation_increment
        )
        found_ticks = len(self.rotation_ticks_peaks)

        if self.expected_tiks_per_rotation == found_ticks:
            logging.info(f"Number of ticks is as expected: {found_ticks}")
            return True
        else:
            logging.warning(
                f"Number of ticks is not as expected: {found_ticks}.\n"
                + f"Expected ticks: {self.expected_tiks_per_rotation}\n"
                + f"Delta: {found_ticks - self.expected_tiks_per_rotation}"
            )
            return False

    def adjust_rotation_increment(self) -> Tuple[np.ndarray, np.ndarray]:
        """Overwrite the method from the parent class to adjust the rotation
        increment and the number of ticks per rotation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The new rotation increment and the number of ticks per rotation.
        """
        ticks_per_rotation = [
            self.get_peaks_in_rotation(start, end)
            for start, end in zip(
                self.rot_blocks_idx["start"],
                self.rot_blocks_idx["end"],
            )
        ]
        new_increments = [
            self.degrees_per_small_rotation / t for t in ticks_per_rotation
        ]

        logging.info(f"New increment example: {new_increments[0]:.3f}")

        return new_increments, ticks_per_rotation

    def get_interpolated_angles(self) -> np.ndarray:
        """Overwrite the method from the parent class to interpolate the
        angles.

        Returns
        -------
        np.ndarray
            The interpolated angles.
        """
        logging.info("Interpolating angles...")

        ticks_with_increment = [
            item
            for i in range(self.number_of_rotations)
            for item in [self.corrected_increments[i]]
            * self.ticks_per_rotation[i]
        ]

        cumulative_sum_to360 = np.cumsum(ticks_with_increment) % self.rot_deg

        interpolated_angles = np.zeros(self.total_clock_time)

        starts, stops = (
            self.rotation_ticks_peaks[:-1],
            self.rotation_ticks_peaks[1:],
        )

        for i, (start, stop) in enumerate(zip(starts, stops)):
            interpolated_angles[start:stop] = np.linspace(
                cumulative_sum_to360[i],
                cumulative_sum_to360[i + 1],
                stop - start,
            )

        return interpolated_angles * -1

    def check_rotation_number_after_interpolation(
        self, start: np.ndarray, end: np.ndarray
    ):
        """Checks that the number of rotations is as expected.

        Raises
        ------
        ValueError
            if the number of start and end of rotations is different
        ValueError
            if the number of rotations is not as expected
        """

        if start.shape[0] != end.shape[0]:
            raise ValueError(
                "Start and end of rotations have different lengths"
            )
        if (
            start.shape[0] != 1
        ):  # The incremental rotation is a unique rotation
            raise ValueError(
                f"Number of rotations is not as expected: {start.shape[0]}"
            )

    def check_number_of_frame_angles(self):
        """Check if the number of rotation angles by frame is equal to the
        number of frames in the image stack.

        Raises
        ------
        ValueError
            if the number of rotation angles by frame is not equal to the
            number of frames in the image stack.
        """
        if len(self.rot_deg_frame) != self.num_frames:
            raise ValueError(
                "Number of rotation angles by frame is not equal to the "
                + "number of frames in the image stack.\n"
                + f"Number of angles: {len(self.rot_deg_frame)}\n"
                + f"Number of frames: {self.num_frames}"
            )

    def check_rotation_number(self, start: np.ndarray, end: np.ndarray):
        """Checks that the number of rotations is as expected.

        Raises
        ------
        ValueError
            if the number of start and end of rotations is different
        ValueError
            if the number of rotations is not as expected
        """

        if start.shape[0] != end.shape[0]:
            raise ValueError(
                "Start and end of rotations have different lengths"
            )
        if start.shape[0] != 1:
            raise ValueError("Number of rotations is not as expected")

    def plot_rotation_angles_and_velocity(self):
        """Plots example rotation angles by line and frame for each speed.
        This plot will be saved in the ``debug_plots`` folder.
        Please inspect it to check that the rotation angles are correctly
        calculated.
        """
        logging.info("Plotting rotation angles...")

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(
            self.frame_start,
            self.rot_deg_frame,
            label="frame angles",
            color="green",
            marker="o",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle("Rotation angles by frame")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        Path(self.config["paths_write"]["debug_plots_folder"]).mkdir(
            parents=True, exist_ok=True
        )
        plt.savefig(
            Path(self.config["paths_write"]["debug_plots_folder"])
            / "rotation_angles.png"
        )

    def plot_rotation_speeds(self):
        #  We don't need to plot the rotation speeds for incremental
        #  rotation pipeline
        pass

    ### ------- Methods unique to IncrementalPipeline ----------------- ###

    def find_center_of_rotation(self) -> Tuple[int, int]:
        """Find the center of rotation by fitting an ellipse to the largest
        blob centers.

        Step 1: Calculate the mean images for each rotation increment.
        Step 2: Find the blobs in the mean images.
        Step 3: Fit an ellipse to the largest blob centers and get its center.

        Returns
        -------
        Tuple[int, int]
            The coordinates center of rotation (x, y).
        """
        logging.info(
            "Fitting an ellipse to the largest blob centers "
            + "to find the center of rotation..."
        )
        mean_images = calculate_mean_images(
            self.image_stack, self.rot_deg_frame
        )

        logging.info("Finding blobs...")
        bd = BlobDetection(self.debugging_plots, self.debug_plots_folder)

        coord_first_blob_of_every_image = bd.get_coords_of_largest_blob(
            mean_images
        )

        # Fit an ellipse to the largest blob centers and get its center
        center_x, center_y, a, b, theta = fit_ellipse_to_points(
            coord_first_blob_of_every_image,
            pixels_in_row=self.num_lines_per_frame,
        )

        if self.debugging_plots:
            plot_ellipse_fit_and_centers(
                coord_first_blob_of_every_image,
                center_x,
                center_y,
                a,
                b,
                theta,
                image_stack=self.image_stack,
                debug_plots_folder=self.debug_plots_folder,
                saving_name="ellipse_fit.png",
            )

        logging.info(
            f"Center of ellipse: ({center_x:.2f}, {center_y:.2f}), "
            f"semi-major axis: {a:.2f}, semi-minor axis: {b:.2f}"
        )
        logging.info(
            "Variation from the center of the image: "
            + f"({center_x - 128:.2f}, {center_y - 128:.2f})"
        )
        logging.info(f"Variation from a perfect circle: {a - b:.2f}")

        #  Raise a warning if the eccentricity is too high
        if np.abs(a - b) > 10:
            logging.warning(
                "The ellipse is too eccentric: "
                + f"{a - b:.2f}; likely due to a bad fit."
            )

        self.all_ellipse_fits = {
            "center_x": center_x,
            "center_y": center_y,
            "a": a,
            "b": b,
            "theta": theta,
        }
        return int(center_x), int(center_y)
