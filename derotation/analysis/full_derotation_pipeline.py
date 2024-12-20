import copy
import logging
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
from fancylog import fancylog
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from tifffile import imsave

from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.load_data.custom_data_loaders import (
    get_analog_signals,
    read_randomized_stim_table,
)


class FullPipeline:
    """DerotationPipeline is a class that derotates an image stack
    acquired with a rotating sample under a microscope.
    """

    ### ----------------- Main pipeline ----------------- ###
    def __init__(self, config_name):
        """DerotationPipeline is a class that derotates an image stack
        acquired with a rotating sample under a microscope.
        In the constructor, it loads the config file, starts the logging
        process, and loads the data.
        It is meant to be used for the full rotation protocol, in which
        the sample is rotated by 360 degrees at various speeds and
        directions.

        Parameters
        ----------
        config_name : str
            Name of the config file without extension.
        """
        self.config = self.get_config(config_name)
        self.start_logging()
        self.load_data()

    def __call__(self):
        """Execute the steps necessary to derotate the image stack
        from start to finish.
        It involves:
        - contrast enhancement
        - processing the analog signals
        - rotating the image stack line by line
        - adding a circular mask to the rotated image stack
        - saving the masked image stack
        """
        self.process_analog_signals()
        rotated_images = self.derotate_frames_line_by_line()
        masked = self.add_circle_mask(rotated_images, self.mask_diameter)
        self.save(masked)
        self.save_csv_with_derotation_data()

    def get_config(self, config_name: str) -> dict:
        """Loads config file from derotation/config folder.
        Please edit it to change the parameters of the analysis.

        Parameters
        ----------
        config_name : str
            Name of the config file without extension.
            Either "full_rotation" or "incremental_rotation".

        Returns
        -------
        dict
            Config dictionary.
        """
        path_config = "derotation/config/" + config_name + ".yml"

        with open(Path(path_config), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return config

    def start_logging(self):
        """Starts logging process using fancylog package.
        Logs saved where specified in the config file.
        """
        path = self.config["paths_write"]["logs_folder"]
        Path(path).mkdir(parents=True, exist_ok=True)
        fancylog.start_logging(
            output_dir=str(path),
            package=sys.modules[__name__.partition(".")[0]],
            filename="derotation",
            verbose=False,
        )

    def load_data(self):
        """Loads data from the paths specified in the config file.
        Data is stored in the class attributes.

        What is loaded:
            * various parameters from config file
            * image stack (tif file)
            * direction and speed of rotation (from randperm file, uses  \
            custom_data_loaders.read_randomized_stim_table)
            * analog signals \
            (from aux file, uses `custom_data_loaders.get_analog_signals`)

        Analog signals are four files, measured in "clock_time":
            * frame clock: on during acquisition of a new frame, off otherwise
            * line clock: on during acquisition of a new line, off otherwise
            * full rotation: when the motor is rotating
            * rotation ticks: peaks at every given increment of rotation

        The data is loaded using the custom_data_loaders module, which are
        specific to the setup used in the lab. Please edit them to load
        data from your setup.
        """
        logging.info("Loading data...")

        self.image_stack = tiff.imread(
            self.config["paths_read"]["path_to_tif"]
        )

        self.num_frames = self.image_stack.shape[0]
        self.num_lines_per_frame = self.image_stack.shape[1]
        self.num_total_lines = self.num_frames * self.num_lines_per_frame
        self.mask_diameter = copy.deepcopy(self.num_lines_per_frame)

        self.direction, self.speed = read_randomized_stim_table(
            self.config["paths_read"]["path_to_randperm"]
        )

        self.number_of_rotations = len(self.direction)

        (
            self.frame_clock,
            self.line_clock,
            self.full_rotation,
            self.rotation_ticks,
        ) = get_analog_signals(
            self.config["paths_read"]["path_to_aux"],
            self.config["channel_names"],
        )

        self.total_clock_time = len(self.frame_clock)

        self.line_use_start = self.config["interpolation"]["line_use_start"]
        self.frame_use_start = self.config["interpolation"]["frame_use_start"]

        self.rotation_increment = self.config["rotation_increment"]
        self.rot_deg = self.config["rot_deg"]

        self.filename_raw = Path(
            self.config["paths_read"]["path_to_tif"]
        ).stem.split(".")[0]
        self.filename = self.config["paths_write"]["saving_name"]

        self.std_coef = self.config["analog_signals_processing"][
            "squared_pulse_k"
        ]
        self.inter_rotation_interval_min_len = self.config[
            "analog_signals_processing"
        ]["inter_rotation_interval_min_len"]

        self.debugging_plots = self.config["debugging_plots"]

        if self.debugging_plots:
            self.debug_plots_folder = Path(
                self.config["paths_write"]["debug_plots_folder"]
            )
            self.debug_plots_folder.mkdir(parents=True, exist_ok=True)

        logging.info(f"Dataset {self.filename_raw} loaded")
        logging.info(f"Filename: {self.filename}")

        #  by default the center of rotation is the center of the image
        self.center_of_rotation = (
            self.num_lines_per_frame // 2,
            self.num_lines_per_frame // 2,
        )
        self.hooks = {}

    ### ----------------- Analog signals processing pipeline ------------- ###
    def process_analog_signals(self):
        """From the analog signals (frame clock, line clock, full rotation,
        rotation ticks) calculates the rotation angles by line and frame.

        It involves:
        - finding rotation ticks peaks
        - identifying the rotation ticks that correspond to
        clockwise and counter clockwise rotations
        - removing various kinds of artifacts that derive from wrong ticks
        - interpolating the angles between the ticks
        - calculating the angles by line and frame

        If debugging_plots is True, it also plots:
        - rotation ticks and the rotation on signal
        - rotation angles by line and frame
        """

        self.rotation_ticks_peaks = self.find_rotation_peaks()

        start, end = self.get_start_end_times_with_threshold(
            self.full_rotation, self.std_coef
        )
        self.rot_blocks_idx = self.correct_start_and_end_rotation_signal(
            start, end
        )
        self.rotation_on = self.create_signed_rotation_array()

        self.drop_ticks_outside_of_rotation()
        self.check_number_of_rotations()

        if not self.is_number_of_ticks_correct():
            (
                self.corrected_increments,
                self.ticks_per_rotation,
            ) = self.adjust_rotation_increment()

        self.interpolated_angles = self.get_interpolated_angles()

        self.remove_artifacts_from_interpolated_angles()

        (
            self.line_start,
            self.line_end,
        ) = self.get_start_end_times_with_threshold(
            self.line_clock, self.std_coef
        )
        (
            self.frame_start,
            self.frame_end,
        ) = self.get_start_end_times_with_threshold(
            self.frame_clock, self.std_coef
        )

        (
            self.rot_deg_line,
            self.rot_deg_frame,
        ) = self.calculate_angles_by_line_and_frame()

        if self.debugging_plots:
            self.plot_rotation_on_and_ticks()
            self.plot_rotation_angles()

        logging.info("✨ Analog signals processed ✨")

    def find_rotation_peaks(self) -> np.ndarray:
        """Finds the peaks of the rotation ticks signal using
        scipy.signal.find_peaks. It filters the peaks using
        the height and distance parameters specified in the config file.

        Returns
        -------
        np.ndarray
            the clock times of the rotation ticks peaks
        """

        logging.info("Finding rotation ticks peaks...")

        height = self.config["analog_signals_processing"][
            "find_rotation_ticks_peaks"
        ]["height"]
        distance = self.config["analog_signals_processing"][
            "find_rotation_ticks_peaks"
        ]["distance"]
        peaks = find_peaks(
            self.rotation_ticks,
            height=height,
            distance=distance,
        )[0]

        return peaks

    @staticmethod
    def get_start_end_times_with_threshold(
        signal: np.ndarray, std_coef: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the start and end times of the on periods of the signal.
        Works for analog signals that have a squared pulse shape.

        Parameters
        ----------
        signal : np.ndarray
            An analog signal.
        std_coef : float
            The factor used to quantify the threshold.

        Returns
        -------
        Tuple(np.ndarray, np.ndarray)
            The start and end times of the on periods of the signal.
        """

        mean = np.mean(signal)
        std = np.std(signal)
        threshold = mean + std_coef * std

        thresholded_signal = np.zeros_like(signal)
        thresholded_signal[signal > threshold] = 1

        start = np.where(np.diff(thresholded_signal) > 0)[0]
        end = np.where(np.diff(thresholded_signal) < 0)[0]

        return start, end

    def correct_start_and_end_rotation_signal(
        self,
        start: np.ndarray,
        end: np.ndarray,
    ) -> dict:
        """Removes artifacts from the start and end times of the on periods
        of the rotation signal. These artifacts appear as very brief off
        periods that are not plausible given the experimental setup.
        The two surrounding on periods are merged.

        Used the inter_rotation_interval_min_len parameter from the config
        file: the minimum length of the time in between two rotations.
        It is important to remove artifacts.

        Parameters
        ----------
        start : np.ndarray
            The start times of the on periods of rotation signal.
        end : np.ndarray
            The end times of the on periods of rotation signal.

        Returns
        -------
        dict
            Corrected start and end times of the on periods of rotation signal.
        """

        logging.info("Cleaning start and end rotation signal...")

        shifted_end = np.roll(end, 1)
        mask = start - shifted_end > self.inter_rotation_interval_min_len
        mask[0] = True  # first rotation is always a full rotation
        shifted_mask = np.roll(mask, -1)
        new_start = start[mask]
        new_end = end[shifted_mask]

        return {"start": new_start, "end": new_end}

    def create_signed_rotation_array(self) -> np.ndarray:
        """Reconstructs an array that has the same length as the full rotation
        signal. It is 0 when the motor is off, and it is 1 or -1 when the motor
        is on, depending on the direction of rotation. 1 is clockwise, -1 is
        counter clockwise.
        Uses the start and end times of the on periods of rotation signal, and
        the direction of rotation to reconstruct the array.

        Returns
        -------
        np.ndarray
            The rotation on signal.
        """

        logging.info("Creating signed rotation array...")
        rotation_on = np.zeros(self.total_clock_time)
        for i, (start, end) in enumerate(
            zip(
                self.rot_blocks_idx["start"],
                self.rot_blocks_idx["end"],
            )
        ):
            rotation_on[start:end] = self.direction[i]

        return rotation_on

    def drop_ticks_outside_of_rotation(self) -> np.ndarray:
        """Drops the rotation ticks that are outside of the rotation periods.

        Returns
        -------
        np.ndarray
            The clock times of the rotation ticks peaks, without the ticks
            outside of the rotation periods.
        """

        logging.info("Dropping ticks outside of the rotation period...")

        len_before = len(self.rotation_ticks_peaks)

        rolled_starts = np.roll(self.rot_blocks_idx["start"], -1)

        # including the interval before the start of the first rotation
        edited_ends = np.insert(self.rot_blocks_idx["end"], 0, 0)
        rolled_starts = np.insert(
            rolled_starts, 0, self.rot_blocks_idx["start"][0]
        )

        # including the end
        rolled_starts[-1] = self.total_clock_time

        inter_roatation_interval = [
            # Effective number of rotations can be different than the one
            # assumed in the config file. Therefore at this stage it is
            # estimated by the number of start and end of rotations
            # calculated from the rotation signal.
            idx
            for i in range(len(edited_ends))
            for idx in range(
                edited_ends[i],
                rolled_starts[i],
            )
        ]

        self.rotation_ticks_peaks = np.delete(
            self.rotation_ticks_peaks,
            np.where(
                np.isin(self.rotation_ticks_peaks, inter_roatation_interval)
            ),
        )

        len_after = len(self.rotation_ticks_peaks)
        logging.info(
            f"Ticks dropped: {len_before - len_after}.\n"
            + f"Ticks remaining: {len_after}"
        )

    def check_number_of_rotations(self):
        """Checks that the number of rotations is as expected.

        Raises
        ------
        ValueError
            if the number of start and end of rotations is different
        ValueError
            if the number of rotations is not as expected
        """

        if (
            self.rot_blocks_idx["start"].shape[0]
            != self.rot_blocks_idx["end"].shape[0]
        ):
            raise ValueError(
                "Start and end of rotations have different lengths"
            )
        if self.rot_blocks_idx["start"].shape[0] != self.number_of_rotations:
            logging.info(
                f"Number of rotations is {self.number_of_rotations}."
                + f"Adjusting to {self.rot_blocks_idx['start'].shape[0]}"
            )
            self.number_of_rotations = self.rot_blocks_idx["start"].shape[0]

        logging.info("Number of rotations is as expected")

    def is_number_of_ticks_correct(self) -> bool:
        """Compares the total number of ticks with the expected number of
        ticks,  which is calculated from the number of rotations and the
        rotation increment.

        Returns
        -------
        bool
            whether the number of ticks is as expected
        """
        self.expected_tiks_per_rotation = (
            self.rot_deg * self.number_of_rotations / self.rotation_increment
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

    def get_peaks_in_rotation(self, start: int, end: int) -> int:
        """Counts the number of ticks in a rotation given the start and end
        times of the rotation.

        Parameters
        ----------
        start : int
            Start clock time of the rotation.
        end : int
            End clock time of the rotation.

        Returns
        -------
        int
            The number of ticks in the rotation.
        """
        return np.where(
            np.logical_and(
                self.rotation_ticks_peaks >= start,
                self.rotation_ticks_peaks <= end,
            )
        )[0].shape[0]

    def adjust_rotation_increment(self) -> Tuple[np.ndarray, np.ndarray]:
        """It calculates the new rotation increment for each rotation, given
        the number of ticks in each rotation. It also outputs the number of
        ticks in each rotation.


        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The new rotation increment for each rotation, and the number of
            ticks in each rotation.
        """

        ticks_per_rotation = [
            self.get_peaks_in_rotation(start, end)
            for start, end in zip(
                self.rot_blocks_idx["start"],
                self.rot_blocks_idx["end"],
            )
        ]
        new_increments = [self.rot_deg / t for t in ticks_per_rotation]

        logging.info(f"New increment example: {new_increments[0]:.3f}")

        return new_increments, ticks_per_rotation

    def get_interpolated_angles(self) -> np.ndarray:
        """Starting from the rotation ticks and knowing the rotation increment,
        it calculates the rotation angles for each clock time.

        Returns
        -------
        np.ndarray
            The rotation angles for each clock time.
        """
        logging.info("Interpolating angles...")

        ticks_with_increment = [
            item
            for i in range(len(self.corrected_increments))
            for item in [self.corrected_increments[i]]
            * self.ticks_per_rotation[i]
        ]

        cumulative_sum_to360 = np.cumsum(ticks_with_increment) % self.rot_deg

        interpolated_angles = np.zeros(self.total_clock_time)

        starts, stops = (
            self.rotation_ticks_peaks[:-1],
            self.rotation_ticks_peaks[1:],
        )

        # interpolate between the ticks
        for i, (start, stop) in enumerate(zip(starts, stops)):
            if cumulative_sum_to360[i + 1] < cumulative_sum_to360[i]:
                cumulative_sum_to360[i] = 0
                cumulative_sum_to360[i + 1] = 0

            interpolated_angles[start:stop] = np.linspace(
                cumulative_sum_to360[i],
                cumulative_sum_to360[i + 1],
                stop - start,
            )

        interpolated_angles = interpolated_angles * self.rotation_on

        return interpolated_angles

    def remove_artifacts_from_interpolated_angles(self):
        """Removes artifacts from the interpolated angles, coming from
        an inconsistency between the number of ticks and cumulative sum.
        These artifacts appear as very brief rotation periods that are not
        plausible given the experimental setup.
        """
        logging.info("Cleaning interpolated angles...")

        self.config["analog_signals_processing"][
            "angle_interpolation_artifact_threshold"
        ]
        thresholded = np.zeros_like(self.interpolated_angles)
        thresholded[np.abs(self.interpolated_angles) > 0.15] = 1
        rotation_start = np.where(np.diff(thresholded) > 0)[0]
        rotation_end = np.where(np.diff(thresholded) < 0)[0]

        self.check_rotation_number_after_interpolation(
            rotation_start, rotation_end
        )

        for start, end in zip(rotation_start[1:], rotation_end[:-1]):
            self.interpolated_angles[end:start] = 0

        self.interpolated_angles[: rotation_start[0]] = 0
        self.interpolated_angles[rotation_end[-1] :] = 0

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
        if start.shape[0] != self.number_of_rotations:
            raise ValueError(
                "Number of rotations is not as expected after interpolation, "
                + f"{start.shape[0]} instead of {self.number_of_rotations}"
            )

    def calculate_angles_by_line_and_frame(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """From the interpolated angles, it calculates the rotation angles
        by line and frame. It can use the start or the end of the line/frame to
        infer the angle.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The rotation angles by line and frame.
        """
        logging.info("Calculating angles by line and frame...")

        inverted_angles = self.interpolated_angles * -1
        line_angles = np.zeros(self.num_total_lines)
        frame_angles = np.zeros(self.num_frames)

        if self.line_use_start:
            line_angles = inverted_angles[self.line_start]
        else:
            line_angles = inverted_angles[self.line_end]

        if self.frame_use_start:
            frame_angles = inverted_angles[self.frame_start]
        else:
            frame_angles = inverted_angles[self.frame_end]

        return line_angles, frame_angles

    def clock_to_latest_line_start(self, clock_time: int) -> int:
        """Get the index of the line that is being scanned at the given clock
        time.

        Parameters
        ----------
        clock_time : int
            The clock time.

        Returns
        -------
        int
            The index of the line
        """
        return np.where(self.line_start < clock_time)[0][-1]

    def clock_to_latest_frame_start(self, clock_time: int) -> int:
        """Get the index of the frame that is being acquired at the given clock
        time.

        Parameters
        ----------
        clock_time : int
            The clock time.

        Returns
        -------
        int
            The index of the frame
        """
        return np.where(self.frame_start < clock_time)[0][-1]

    def clock_to_latest_rotation_start(self, clock_time: int) -> int:
        """Get the index of the latest rotation that happened.

        Parameters
        ----------
        clock_time : int
            The clock time.

        Returns
        -------
        int
            The index of the latest rotation
        """
        return np.where(self.rot_blocks_idx["start"] < clock_time)[0][-1]

    def plot_rotation_on_and_ticks(self):
        """Plots the rotation ticks and the rotation on signal.
        This plot will be saved in the debug_plots folder.
        Please inspect it to check that the rotation ticks are correctly
        placed during the times in which the motor is rotating.
        """

        logging.info("Plotting rotation ticks and rotation on signal...")

        fig, ax = plt.subplots(1, 1, figsize=(20, 5))

        ax.scatter(
            self.rotation_ticks_peaks,
            self.rotation_on[self.rotation_ticks_peaks],
            label="rotation ticks",
            marker="o",
            alpha=0.5,
            color="orange",
        )
        ax.plot(
            self.rotation_on,
            label="rotation on",
            color="navy",
        )

        ax.plot(
            self.full_rotation / np.max(self.full_rotation),
            label="rotation ticks",
            color="black",
            alpha=0.2,
        )

        ax.set_title("Rotation ticks and rotation on signal")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.savefig(
            self.debug_plots_folder / "rotation_ticks_and_rotation_on.png"
        )

    def plot_rotation_angles(self):
        """Plots example rotation angles by line and frame for each speed.
        This plot will be saved in the debug_plots folder.
        Please inspect it to check that the rotation angles are correctly
        calculated.
        """
        logging.info("Plotting rotation angles...")

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        speeds = set(self.speed)

        last_idx_for_each_speed = [
            np.where(self.speed == s)[0][-1] for s in speeds
        ]
        last_idx_for_each_speed = sorted(last_idx_for_each_speed)

        for i, id in enumerate(last_idx_for_each_speed):
            col = i // 2
            row = i % 2

            ax = axs[col, row]

            rotation_starts = self.rot_blocks_idx["start"][id]
            rotation_ends = self.rot_blocks_idx["end"][id]

            start_line_idx = self.clock_to_latest_line_start(rotation_starts)
            end_line_idx = self.clock_to_latest_line_start(rotation_ends)

            start_frame_idx = self.clock_to_latest_frame_start(rotation_starts)
            end_frame_idx = self.clock_to_latest_frame_start(rotation_ends)

            ax.scatter(
                self.line_start[start_line_idx:end_line_idx],
                self.rot_deg_line[start_line_idx:end_line_idx],
                label="line angles",
                color="orange",
                marker="o",
            )

            ax.scatter(
                self.frame_start[start_frame_idx:end_frame_idx],
                self.rot_deg_frame[start_frame_idx:end_frame_idx],
                label="frame angles",
                color="green",
                marker="o",
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.set_title(
                f"Speed: {self.speed[id]},"
                + f" direction: {'cw' if self.direction[id] == 1 else 'ccw'}"
            )

        fig.suptitle("Rotation angles by line and frame")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        plt.savefig(self.debug_plots_folder / "rotation_angles.png")

    ### ----------------- Derotation ----------------- ###

    def plot_max_projection_with_center(self):
        """Plots the maximum projection of the image stack with the center
        of rotation.
        This plot will be saved in the debug_plots folder.
        Please inspect it to check that the center of rotation is correctly
        placed.
        """
        logging.info("Plotting max projection with center...")

        max_projection = np.max(self.image_stack, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.imshow(max_projection, cmap="gray")
        ax.scatter(
            self.center_of_rotation[0],
            self.center_of_rotation[1],
            color="red",
            marker="x",
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.axis("off")

        plt.savefig(self.debug_plots_folder / "max_projection_with_center.png")

    def derotate_frames_line_by_line(self) -> np.ndarray:
        """Wrapper for the function `derotate_an_image_array_line_by_line`.
        Before calling the function, it finds the F0 image offset with
        `find_image_offset`.

        Returns
        -------
        np.ndarray
            The rotated image stack.
        """
        logging.info("Starting derotation by line...")

        if self.debugging_plots:
            self.plot_max_projection_with_center()

        offset = self.find_image_offset(self.image_stack[0])

        rotated_image_stack = derotate_an_image_array_line_by_line(
            self.image_stack,
            self.rot_deg_line,
            blank_pixels_value=offset,
            center=self.center_of_rotation,
            plotting_hook_line_addition=self.hooks.get(
                "plotting_hook_line_addition"
            ),
            plotting_hook_image_completed=self.hooks.get(
                "plotting_hook_image_completed"
            ),
        )

        logging.info("✨ Image stack rotated ✨")
        return rotated_image_stack

    @staticmethod
    def find_image_offset(img):
        """Find the "F0", also called "image offset" for a given image.

        Explanations
        ------------
        What is the image offset?
        The PMT (photo-multiplier tube) adds an arbitrary offset to the
        image that corresponds to 0 photons received. We can use a Gaussian
        Mixture Model to find this offset by assuming that it will be the
        smallest mean of the Gaussian components.

        Why do we need to find it?
        When we rotate the image, the pixels of the image that are not
        sampled will be filled with the offset is order to correctly express
        "0 photons received".

        Parameters
        ----------
        img : np.ndarray
            The image for which you want to find the offset.

        Returns
        -------
        float
            The offset.
        """
        gm = GaussianMixture(n_components=7, random_state=0).fit(
            img[0].reshape(-1, 1)
        )
        offset = np.min(gm.means_)
        return offset

    ### ----------------- Saving ----------------- ###
    @staticmethod
    def add_circle_mask(
        image_stack: np.ndarray,
        diameter: int = 256,
    ) -> np.ndarray:
        """Adds a circular mask to the rotated image stack. It is useful
        to hide the portions of the image that are not sampled equally
        during the rotation.
        If a diameter is specified, the image stack is cropped to match
        the diameter. The mask is then added to the cropped image stack,
        and the cropped image stack is padded to match the original size.
        This is important when the images are registered to correct from
        motion artifacts.

        Parameters
        ----------
        image_stack : np.ndarray
            The image stack that you want to mask.
        diameter : int, optional
            The diameter of the circular mask, by default 256

        Returns
        -------
        np.ndarray
            The masked image stack.
        """
        img_height = image_stack.shape[1]

        #  crop the image to match the new size
        if diameter != img_height:
            image_stack = image_stack[
                :,
                int((img_height - diameter) / 2) : int(
                    (img_height + diameter) / 2
                ),
                int((img_height - diameter) / 2) : int(
                    (img_height + diameter) / 2
                ),
            ]

        xx, yy = np.mgrid[:diameter, :diameter]

        circle = (xx - diameter / 2) ** 2 + (yy - diameter / 2) ** 2
        mask = circle < (diameter / 2) ** 2

        img_min = np.nanmin(image_stack)

        masked_img_array = []
        for img in image_stack:
            masked_img_array.append(np.where(mask, img, img_min))

        # pad the image to match the original size
        if diameter != img_height:
            delta = img_height - diameter
            masked_img_array = np.pad(
                masked_img_array,
                ((0, 0), (int(delta / 2), int(delta / 2)), (0, 0)),
                "constant",
                constant_values=img_min,
            )
            masked_img_array = np.pad(
                masked_img_array,
                ((0, 0), (0, 0), (int(delta / 2), int(delta / 2))),
                "constant",
                constant_values=img_min,
            )

        return np.array(masked_img_array)

    def save(self, masked: np.ndarray):
        """Saves the masked image stack in the saving folder specified in the
        config file.

        Parameters
        ----------
        masked : np.ndarray
            The masked derotated image stack.
        """
        path = self.config["paths_write"]["derotated_tiff_folder"]
        Path(path).mkdir(parents=True, exist_ok=True)

        imsave(
            path + self.config["paths_write"]["saving_name"] + ".tif",
            np.array(masked),
        )
        logging.info(f"Masked image saved in {path}")

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

        df["direction"] = np.nan * np.ones(len(df))
        df["speed"] = np.nan * np.ones(len(df))
        df["rotation_count"] = np.nan * np.ones(len(df))

        rotation_counter = 0
        adding_roatation = False
        for row_idx in range(len(df)):
            if np.abs(df.loc[row_idx, "rotation_angle"]) > 0.0:
                adding_roatation = True
                df.loc[row_idx, "direction"] = self.direction[rotation_counter]
                df.loc[row_idx, "speed"] = self.speed[rotation_counter]
                df.loc[row_idx, "rotation_count"] = rotation_counter
            if (
                rotation_counter < self.number_of_rotations - 1
                and adding_roatation
                and np.abs(df.loc[row_idx + 1, "rotation_angle"]) == 0.0
            ):
                rotation_counter += 1
                adding_roatation = False

        Path(self.config["paths_write"]["derotated_tiff_folder"]).mkdir(
            parents=True, exist_ok=True
        )

        df.to_csv(
            self.config["paths_write"]["derotated_tiff_folder"]
            + self.config["paths_write"]["saving_name"]
            + ".csv",
            index=False,
        )
