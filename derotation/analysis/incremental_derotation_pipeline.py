import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from skimage.feature import blob_log
from tqdm import tqdm

from derotation.analysis.fit_ellipse import (
    fit_ellipse_to_points,
    plot_ellipse_fit_and_centers,
)
from derotation.analysis.full_derotation_pipeline import FullPipeline


class IncrementalPipeline(FullPipeline):
    """Derotate the image stack that was acquired using the incremental
    rotation method.
    """

    ### ------- Methods to overwrite from the parent class ------------ ###
    def __init__(self, *args, **kwargs):
        """Derotate the image stack that was acquired using the incremental
        rotation method.
        As a child of FullPipeline, it inherits all the attributes and
        methods from it and adds additional logic to register the images.
        Here in the constructor we specify the degrees for each incremental
        rotation and the number of rotations.
        """
        super().__init__(*args, **kwargs)

        self.small_rotations = 10
        self.number_of_rotations = self.rot_deg // self.small_rotations

    def __call__(self):
        """Overwrite the __call__ method from the parent class to derotate
        the image stack acquired using the incremental rotation method.
        After processing the analog signals, the image stack is rotated by
        frame and then registered using phase cross correlation.
        """
        super().process_analog_signals()
        derotated_images = self.deroatate_by_frame()
        masked_unregistered = self.add_circle_mask(derotated_images)

        mean_images = self.calculate_mean_images(masked_unregistered)
        target_image = self.get_target_image(masked_unregistered)
        shifts = self.get_shifts_using_phase_cross_correlation(
            mean_images, target_image
        )
        x_fitted, y_fitted = self.polynomial_fit(shifts)
        registered_images = self.register_rotated_images(
            masked_unregistered, x_fitted, y_fitted
        )

        self.new_diameter = self.num_lines_per_frame - 2 * max(
            max(np.abs(shifts["x"])), max(np.abs(shifts["y"]))
        )
        masked = self.add_circle_mask(registered_images, self.new_diameter)

        self.save(masked)
        self.save_csv_with_derotation_data()

        self.center_of_rotation = self.find_center_of_rotation()

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
        new_increments = [self.small_rotations / t for t in ticks_per_rotation]

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

    def plot_rotation_angles(self):
        """Plots example rotation angles by line and frame for each speed.
        This plot will be saved in the debug_plots folder.
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
        plt.close()

    def save_csv_with_derotation_data(self):
        """Saves a csv file with the rotation angles by line and frame,
        and the rotation on signal.
        It is saved in the saving folder specified in the config file.
        """
        df = pd.DataFrame(
            columns=[
                "frame",
                "rotation_angle",
                "clock",
            ]
        )

        df["frame"] = np.arange(self.num_frames)
        df["rotation_angle"] = self.rot_deg_frame[: self.num_frames]
        df["clock"] = self.frame_start[: self.num_frames]

        Path(self.config["paths_write"]["derotated_tiff_folder"]).mkdir(
            parents=True, exist_ok=True
        )

        df.to_csv(
            self.config["paths_write"]["derotated_tiff_folder"]
            + self.config["paths_write"]["saving_name"]
            + ".csv",
            index=False,
        )

    ### ------- Methods unique to IncrementalPipeline ----------------- ###
    def deroatate_by_frame(self) -> np.ndarray:
        """Rotate the image stack by frame.

        Returns
        -------
        np.ndarray
            Description of returned object.
        """
        logging.info("Starting derotation by frame...")
        min_value_img = np.min(self.image_stack)
        new_rotated_image_stack = (
            np.ones_like(self.image_stack) * min_value_img
        )

        for idx, frame in tqdm(
            enumerate(self.image_stack), total=self.num_frames
        ):
            rotated_img = rotate(
                frame,
                self.rot_deg_frame[idx],
                reshape=False,
                order=0,
                mode="constant",
            )
            rotated_img = np.where(
                rotated_img == 0, min_value_img, rotated_img
            )

            new_rotated_image_stack[idx] = rotated_img

        logging.info("Finished rotating the image stack")

        return new_rotated_image_stack

    @staticmethod
    def get_target_image(rotated_image_stack: np.ndarray) -> np.ndarray:
        """Get the target image for phase cross correlation. This is the mean
        of the first 100 images.

        Parameters
        ----------
        rotated_image_stack : np.ndarray
            The rotated image stack.

        Returns
        -------
        np.ndarray
            The target image.
        """
        return np.mean(rotated_image_stack[:100], axis=0)

    def get_shifts_using_phase_cross_correlation(
        self, mean_images: list, target_image: np.ndarray
    ) -> Dict[str, list]:
        """Get the shifts (i.e. the number of pixels that the image needs to
        be shifted in order to be registered) using phase cross correlation.

        Parameters
        ----------
        mean_images : list
            The list of mean images for each rotation increment.
        target_image : np.ndarray
            The target image.

        Returns
        -------
        Dict[str, list]
            The shifts in x and y.
        """

        logging.info("Calculating shifts using phase cross correlation...")
        shifts: Dict[str, list] = {"x": [], "y": []}
        image_center = self.num_lines_per_frame / 2
        for offset_image in mean_images:
            image_product = (
                np.fft.fft2(target_image) * np.fft.fft2(offset_image).conj()
            )
            cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
            peaks = np.unravel_index(np.argmax(cc_image), cc_image.shape)

            shift = np.asarray(peaks) - image_center
            shifts["x"].append(int(shift[0]))
            shifts["y"].append(int(shift[1]))

        return shifts

    def polynomial_fit(self, shifts: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a polynomial to the shifts in order to get a smooth function
        that can be used to register the images.

        Parameters
        ----------
        shifts : dict
            The shifts in x and y.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The polynomial fit for x and y.
        """
        logging.info("Fitting polynomial to shifts...")

        shifts["x"].insert(0, 0)
        shifts["y"].insert(0, 0)

        angles_range = np.arange(0, 360, 10)
        x = shifts["x"]
        y = shifts["y"]

        x_fitted = np.polyfit(angles_range, x, 6)
        y_fitted = np.polyfit(angles_range, y, 6)

        return x_fitted, y_fitted

    def register_rotated_images(
        self,
        rotated_image_stack: np.ndarray,
        x_fitted: np.ndarray,
        y_fitted: np.ndarray,
    ) -> np.ndarray:
        """Register the rotated images using the polynomial fit.

        Parameters
        ----------
        rotated_image_stack : np.ndarray
            The rotated image stack.
        x_fitted : np.ndarray
            The polynomial fit for x.
        y_fitted : np.ndarray
            The polynomial fit for y.

        Returns
        -------
        np.ndarray
            The registered image stack.
        """

        logging.info("Registering rotated images...")
        registered_images = []
        for i, img in enumerate(rotated_image_stack):
            angle = self.rot_deg_frame[i]
            x = np.polyval(x_fitted, angle)
            y = np.polyval(y_fitted, angle)
            shift = (int(x), int(y))
            registered_images.append(np.roll(img, shift=shift, axis=(0, 1)))

        registered_images = np.array(registered_images)

        return registered_images

    def find_center_of_rotation(self, method="track") -> Tuple[int, int]:
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
        mean_images = self.calculate_mean_images(self.image_stack)

        logging.info("Finding blobs...")
        if method == "track":
            coord_first_blob_of_every_image = self.get_coords_of_tracked_blob(
                mean_images
            )
        if method == "largest":
            coord_first_blob_of_every_image = self.get_coords_of_largest_blob(
                mean_images
            )

        # Fit an ellipse to the largest blob centers and get its center
        center_x, center_y, a, b, theta = fit_ellipse_to_points(
            coord_first_blob_of_every_image
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

        #  if the variation from a perfect circle is too high, log a warning and store it in a variable
        if np.abs(a - b) > 10:
            logging.warning(
                "The variation from a perfect circle is too high: "
                + f"{a - b:.2f}; likely due to a bad fit."
            )
            self.good_ellipse_fit = False
        else:
            self.good_ellipse_fit = True

        self.all_ellipse_fits = {
            "center_x": center_x,
            "center_y": center_y,
            "a": a,
            "b": b,
            "theta": theta,
        }
        return int(center_x), int(center_y)

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
            blob_log(img, max_sigma=12, min_sigma=7, threshold=0.95, overlap=0)
            for img in tqdm(image_stack)
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

    def get_coords_of_tracked_blob(
        self, image_stack: np.ndarray
    ) -> np.ndarray:
        """Track a blob across frames based on circular motion prediction.

        Parameters
        ----------
        image_stack : np.ndarray
            The image stack.

        Returns
        -------
        np.ndarray
            The coordinates of the tracked blob in each image.
        """

        def predict_next_position(center, prev_coord, radius, angle_increment):
            """Predict the next blob position on the circle based on the previous position."""
            # Calculate the angle based on the previous position relative to the center
            delta_x, delta_y = prev_coord - center
            prev_angle = np.arctan2(delta_y, delta_x)

            # Increment the angle to predict the next position
            next_angle = prev_angle + angle_increment

            # Calculate the new coordinates
            x = center[0] + radius * np.cos(next_angle)
            y = center[1] + radius * np.sin(next_angle)

            # # log everything for debugging
            # logging.info(f"Center: {center}")
            # logging.info(f"Previous coordinate: {prev_coord}")
            # logging.info(f"Radius: {radius}")
            # logging.info(f"Angle increment: {angle_increment}")
            # logging.info(f"Previous angle: {prev_angle}")
            # logging.info(f"Next angle: {next_angle}")
            # logging.info(f"Predicted coordinate: ({x}, {y})")

            return np.array([x, y])

        def find_closest_blob(predicted_coord, current_blobs, max_distance):
            """Find the closest blob to the predicted coordinate within a threshold."""
            distances = np.linalg.norm(
                current_blobs[:, :2] - predicted_coord, axis=1
            )
            min_distance_idx = np.argmin(distances)
            logging.info(
                f"Distance to closest blob: {distances[min_distance_idx]}"
            )
            if distances[min_distance_idx] <= max_distance:
                return current_blobs[min_distance_idx]
            else:
                return None  # No blob within the threshold

        def find_most_isolated(numbers):
            """
            Finds the most isolated number in a list, defined as the number
            with the largest minimum distance to all other numbers.

            Parameters:
            numbers (list or array-like): A list of numbers.

            Returns:
            float: The most isolated number.
            """
            # Convert to a numpy array for efficient computation
            numbers = np.array(numbers)

            # Compute pairwise distances
            distances = np.abs(numbers[:, None] - numbers)

            # Exclude self-distances by setting the diagonal to infinity
            np.fill_diagonal(distances, np.inf)

            # Find the minimum distance to any other number for each number
            min_distances = distances.min(axis=1)

            # Identify the number with the largest minimum distance
            most_isolated_index = np.argmax(min_distances)
            return most_isolated_index

        blobs_in_frames = [
            blob_log(img, max_sigma=12, min_sigma=7, threshold=0.95, overlap=0)
            for img in tqdm(image_stack)
        ]

        #  blobs list description:
        # blobs[i] is an array of shape (n, 3) where n is the number of blobs
        # detected in image i. The columns are (y, x, r) where (y, x) is the
        # center of the blob and r is the radius.
        #  blobs length is the number of images in the stack

        # Initialize parameters
        tracked_coords = []
        prev_coord = None
        angle_increment = np.deg2rad(10)  # Approximate rotation per frame
        angle = np.deg2rad(10)  # Initial angle
        center = self.center_of_rotation
        radius = None  # To be calculated from the first blob

        max_distance = 150  # Threshold for discarding blobs (adjust as needed)
        tollerance = 0
        radius_tollerance = 15
        min_distance = 70

        for i, blobs_in_a_frame in enumerate(blobs_in_frames):
            if len(blobs_in_a_frame) > 0:
                if i == 0:
                    distances = np.linalg.norm(
                        blobs_in_a_frame[:, :2] - center, axis=1
                    )
                    logging.info(distances)
                    #  only consider blobs further than 50px from the center
                    criteria = (distances > min_distance) & (
                        distances < max_distance
                    )
                    blobs_in_a_frame = blobs_in_a_frame[criteria]

                    most_isolated_index = find_most_isolated(
                        distances[criteria]
                    )
                    most_isolated_blob = blobs_in_a_frame[most_isolated_index]
                    prev_coord = most_isolated_blob[:2]

                    # largest_blob = blobs_in_a_frame[np.argmax(blobs_in_a_frame[:, 2])]
                    # logging.info(largest_blob)
                    # prev_coord = largest_blob[:2]
                    radius = np.linalg.norm(
                        prev_coord - center
                    )  # Calculate initial radius
                    logging.info(f"Initial radius: {radius}")
                    tracked_coords.append(prev_coord.astype(int))
                else:
                    # Filter blobs based on distance from the center
                    distances = np.linalg.norm(
                        blobs_in_a_frame[:, :2] - center, axis=1
                    )
                    #  only consider blobs within +- 30 pix from the radius
                    correct_radius = (
                        np.abs(distances - radius) < radius_tollerance
                    )

                    # # if prev_coord is not None:
                    # # distance_from_previous = np.linalg.norm(blobs_in_a_frame[:, :2] - prev_coord, axis=1)
                    # # correct_distance = distance_from_previous < (max_distance + tollerance)
                    # # blobs_in_a_frame = blobs_in_a_frame[correct_radius & correct_distance]
                    # # else:

                    blobs_in_a_frame = blobs_in_a_frame[correct_radius]

                    # #  filter based on most isolated blob
                    # distances = np.linalg.norm(blobs_in_a_frame[:, :2] - center, axis=1)
                    # most_isolated_index = find_most_isolated(distances)
                    # most_isolated_blob = blobs_in_a_frame[most_isolated_index]
                    # blobs_in_a_frame = np.array([most_isolated_blob])

                    if len(blobs_in_a_frame) == 0:
                        # prev_coord = predict_next_position(center, prev_coord, radius, angle_increment)
                        # logging.info(f"Predicted position: {prev_coord}")
                        tollerance += max_distance
                        tracked_coords.append([np.nan, np.nan])
                        logging.warning(f"No blobs were found in image {i}")
                    elif len(blobs_in_a_frame) == 1:
                        prev_coord = blobs_in_a_frame[0][:2]
                        tracked_coords.append(prev_coord.astype(int))
                        logging.info(
                            f"Blob found in image {i}, position: {prev_coord}"
                        )
                        tollerance = 0
                    else:
                        # #  pick the closest blob to the previous position
                        # closest_blob = find_closest_blob(prev_coord, blobs_in_a_frame, max_distance)
                        # if closest_blob is not None:
                        #     prev_coord = closest_blob[:2]
                        #     tracked_coords.append(prev_coord.astype(int))
                        #     logging.info(f"Blob found in image {i}, position: {prev_coord}")
                        #     tollerance = 0
                        # else:
                        #     prev_coord = predict_next_position(center, prev_coord, radius, angle_increment)
                        #     tracked_coords.append([np.nan, np.nan])
                        #     logging.warning(f"No blobs were found in image {i}")

                        # #  pick the largest blob
                        # largest_blob = blobs_in_a_frame[np.argmax(blobs_in_a_frame[:, 2])]
                        # prev_coord = largest_blob[:2]

                        # #  pick the blob less far from the radius
                        # distances = np.linalg.norm(blobs_in_a_frame[:, :2] - center, axis=1)
                        # closest_blob = blobs_in_a_frame[np.argmin(distances)]
                        # prev_coord = closest_blob[:2]
                        # tracked_coords.append(prev_coord.astype(int))
                        # logging.info(f"Blob found in image {i}, position: {prev_coord}")
                        # tollerance = 0

                        # #  pick the furthest blob
                        # distances = np.linalg.norm(blobs_in_a_frame[:, :2] - center, axis=1)
                        # furthest_index = np.argmax(distances)
                        # furthest_blob = blobs_in_a_frame[furthest_index]
                        # prev_coord = furthest_blob[:2]
                        # tracked_coords.append(prev_coord.astype(int))
                        # logging.info(f"Blob found in image {i}, position: {prev_coord}")

                        # take all blobs
                        for blob in blobs_in_a_frame:
                            tracked_coords.append(blob[:2].astype(int))
                            logging.info(
                                f"Blob found in image {i}, position: {blob[:2]}"
                            )

                        # #  pick the most isolated blob
                        # distances = np.linalg.norm(blobs_in_a_frame[:, :2] - center, axis=1)
                        # most_isolated_index = find_most_isolated(distances)
                        # most_isolated_blob = blobs_in_a_frame[most_isolated_index]
                        # prev_coord = most_isolated_blob[:2]
                        # tracked_coords.append(prev_coord.astype(int))
                        # logging.info(f"Blob found in image {i}, position: {prev_coord}")

                        # #  nan
                        # tracked_coords.append([np.nan, np.nan])
                        # logging.warning(f"No blobs were found in image {i}")

                        #  pick the closest blob to the predicted position
                        # predicted = predict_next_position(center, prev_coord, radius, angle_increment)
                        # closest_blob = find_closest_blob(predicted, blobs_in_a_frame, max_distance)

                        # if closest_blob is not None:
                        #     prev_coord = closest_blob[:2]
                        #     tracked_coords.append(prev_coord.astype(int))
                        #     logging.info(f"Blob found in image {i}, position: {prev_coord}")
                        #     tollerance = 0
                        # else:
                        #     prev_coord = predict_next_position(center, prev_coord, radius, angle_increment)
                        #     tracked_coords.append([np.nan, np.nan])
                        #     logging.warning(f"No blobs were found in image {i}")

                    # Update angle for the next prediction
                    angle += angle_increment
            else:
                # Handle case where no blobs are found
                tracked_coords.append([np.nan, np.nan])
                tollerance += max_distance
                logging.warning(f"No blobs were found in image {i}")

        # Invert x, y order
        tracked_coords = [(coord[1], coord[0]) for coord in tracked_coords]

        # Plot blobs on top of every frame
        if self.debugging_plots:
            self.plot_blob_detection(blobs_in_frames, image_stack)

        logging.info(f"Tracked coordinates: {tracked_coords}")

        return np.asarray(tracked_coords)

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
            a.set_title(f"{i*5} degrees")
            a.axis("off")

            for j, blob in enumerate(blobs[i][:4]):
                y, x, r = blob
                c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
                a.add_patch(c)
                a.text(x, y, str(j), color="red")

        # save the plot
        plt.savefig(self.debug_plots_folder / "blobs.png")
        plt.close()
