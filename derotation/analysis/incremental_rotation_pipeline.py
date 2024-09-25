import copy
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from skimage.feature import blob_log
from tqdm import tqdm

from derotation.analysis.full_rotation_pipeline import FullPipeline


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
        rotated_images = self.roatate_by_frame()
        masked_unregistered = self.add_circle_mask(rotated_images)

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

    def calculate_mean_images(self, rotated_image_stack: np.ndarray) -> list:
        """Calculate the mean images for each rotation angle. This required
        to calculate the shifts using phase cross correlation.

        Parameters
        ----------
        rotated_image_stack : np.ndarray
            The rotated image stack.

        Returns
        -------
        list
            The list of mean images.
        """
        logging.info("Calculating mean images...")

        #  correct for a mismatch in the total number of frames
        #  and the number of angles, given by instrument error
        angles_subset = copy.deepcopy(self.rot_deg_frame[2:])
        # also there is a bias on the angles
        angles_subset += -0.1
        rounded_angles = np.round(angles_subset, 2)

        mean_images = []
        for i in np.arange(10, 360, 10):
            images = rotated_image_stack[rounded_angles == i]
            mean_image = np.mean(images, axis=0)

            mean_images.append(mean_image)

        return mean_images

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
    def roatate_by_frame(self) -> np.ndarray:
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

    def find_center_of_rotation(self):
        logging.info("Finding the center of rotation...")

        mean_images = self.calculate_mean_images(self.image_stack)
        sample_number = len(mean_images)
        subgroup = mean_images[:sample_number]

        logging.info("Finding blobs...")
        blobs = [
            blob_log(img, max_sigma=12, min_sigma=7, threshold=0.95, overlap=0)
            for img in tqdm(subgroup)
        ]

        # sort blobs by size
        blobs = [
            blobs[i][blobs[i][:, 2].argsort()] for i in range(sample_number)
        ]

        # plot blobs on top of every frame
        if self.debugging_plots:
            self.plot_blob_detection(blobs, subgroup)

        coord_first_blob_of_every_image = [
            blobs[i][0][:2].astype(int) for i in range(6)
        ]
        centre = self.find_centre_with_bisector(
            coord_first_blob_of_every_image
        )

        logging.info(f"Centre of rotation: {centre}")
        logging.info(
            "Difference with the expected centre:"
            + f"{np.array([256 // 2, 256 // 2]) - centre}"
        )
        return centre

    def plot_blob_detection(self, blobs, subgroup):
        fig, ax = plt.subplots(4, 3, figsize=(10, 10))
        for i, a in tqdm(enumerate(ax.flatten())):
            a.imshow(subgroup[i])
            a.set_title(f"{i*5} degrees")
            a.axis("off")

            for j, blob in enumerate(blobs[i][:4]):
                y, x, r = blob
                c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
                a.add_patch(c)
                a.text(x, y, str(j), color="red")

        # save the plot
        plt.savefig(
            Path(self.config["paths_write"]["debug_plots_folder"])
            / "blobs.png"
        )

    def find_centre_with_bisector(self, coords):
        # Find the line bisector between coords1 and coords2
        def find_bisector_between_two_coords(coords1, coords2):
            x1, y1 = coords1
            x2, y2 = coords2
            m1 = -((x1 - x2) / (y1 - y2))
            mx1, my1 = (x1 + x2) / 2, (y1 + y2) / 2
            c1 = my1 - (m1 * mx1)
            return m1, c1

        def find_the_center(m1, c1, m2, c2):
            cx = (c2 - c1) / (m1 - m2)
            cy = (m1 * cx) + c1
            return cx, cy

        # coords is an array of N coordinates
        # for each combination of coordinates, find the bisector
        # and the corresponding center
        centers = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                m1, c1 = find_bisector_between_two_coords(coords[i], coords[j])
                for k in range(j + 1, len(coords)):
                    m2, c2 = find_bisector_between_two_coords(
                        coords[j], coords[k]
                    )
                    cx, cy = find_the_center(m1, c1, m2, c2)
                    centers.append((cx, cy))

        centers = np.array(centers)
        # centers to integers
        centers = np.round(centers).astype(int)

        mean_center = np.mean(centers, axis=0)
        median_center = np.median(centers, axis=0)
        mode_center = np.mean(centers, axis=0)
        logging.info(
            f"Mean center: {mean_center}, Median center: {median_center},"
            + f"Mode center: {mode_center}"
        )

        # plot all centers, plot mean, median, mode
        if self.debugging_plots:
            self.plot_center_distribution(
                centers, mean_center, median_center, mode_center
            )

        # mode to integer
        mode_center = mode_center.astype(int)
        return mode_center

    def plot_center_distribution(
        self, centers, mean_center, median_center, mode_center
    ):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(self.image_stack[0])
        for center in centers:
            ax.scatter(center[0], center[1], color="red", marker="x")
        #  plot theoretical center in green
        ax.scatter(
            len(self.image_stack[0]) // 2,
            len(self.image_stack[0]) // 2,
            color="green",
            marker="x",
        )
        ax.scatter(mean_center[0], mean_center[1], color="green", label="Mean")
        ax.scatter(
            median_center[0], median_center[1], color="blue", label="Median"
        )
        ax.scatter(
            mode_center[0], mode_center[1], color="purple", label="Mode"
        )
        ax.axis("off")
        ax.legend()

        plt.savefig(
            Path(self.config["paths_write"]["debug_plots_folder"])
            / "center_distribution.png"
        )
