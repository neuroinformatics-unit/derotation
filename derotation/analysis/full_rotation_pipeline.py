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
from skimage.exposure import rescale_intensity
from tifffile import imsave

from derotation.derotate_by_line import rotate_an_image_array_line_by_line
from derotation.load_data.custom_data_loaders import (
    get_analog_signals,
    read_randomized_stim_table,
)


class FullPipeline:
    """DerotationPipeline is a class that derotates an image stack
    acquired with a rotating sample under a microscope.
    """

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
        self.contrast_enhancement()
        self.process_analog_signals()
        rotated_images = self.rotate_frames_line_by_line()
        masked = self.add_circle_mask(rotated_images)
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
        Logs saved whenre specified in the config file.
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
        sepecific to the setup used in the lab. Please edit them to load
        data from your setup.
        """
        logging.info("Loading data...")

        self.image_stack = tiff.imread(
            self.config["paths_read"]["path_to_tif"]
        )

        self.num_frames = self.image_stack.shape[0]
        self.num_lines_per_frame = self.image_stack.shape[1]
        self.num_total_lines = self.num_frames * self.num_lines_per_frame

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
        self.adjust_increment = self.config["adjust_increment"]

        self.filename_raw = Path(
            self.config["paths_read"]["path_to_tif"]
        ).stem.split(".")[0]
        self.filename = self.config["paths_write"]["saving_name"]

        self.k = self.config["analog_signals_processing"]["squared_pulse_k"]
        self.inter_rotation_interval_min_len = self.config[
            "analog_signals_processing"
        ]["inter_rotation_interval_min_len"]

        self.debugging_plots = self.config["debugging_plots"]

        logging.info(f"Dataset {self.filename_raw} loaded")
        logging.info(f"Filename: {self.filename}")

    def contrast_enhancement(self):
        """Applies contrast enhancement to the image stack.
        It is useful to visualize the image stack before derotation.
        """
        logging.info("Applying contrast enhancement...")

        self.image_stack = np.array(
            [
                self.contrast_enhancement_single_image(
                    img, self.config["contrast_enhancement"]
                )
                for img in self.image_stack
            ]
        )

    @staticmethod
    def contrast_enhancement_single_image(
        img: np.ndarray, saturated_percentage=0.35
    ) -> np.ndarray:
        """Applies contrast enhancement to a single image.
        It is useful to visualize the image stack before derotation.

        Parameters
        ----------
        img : np.ndarray
            The image to enhance.
        saturated_percentage : float, optional
            The percentage of saturated pixels, by default 0.35

        Returns
        -------
        np.ndarray
            The enhanced image.
        """
        v_min, v_max = np.percentile(
            img, (saturated_percentage, 100 - saturated_percentage)
        )
        return rescale_intensity(img, in_range=(v_min, v_max))

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
            self.full_rotation, self.k
        )
        self.rot_blocks_idx = self.correct_start_and_end_rotation_signal(
            start, end
        )
        self.rotation_on = self.create_signed_rotation_array()

        self.drop_ticks_outside_of_rotation()

        self.check_number_of_rotations()
        if not self.is_number_of_ticks_correct() and self.adjust_increment:
            (
                self.corrected_increments,
                self.ticks_per_rotation,
            ) = self.adjust_rotation_increment()

        self.interpolated_angles = self.get_interpolated_angles()

        self.remove_artifacts_from_interpolated_angles()

        (
            self.line_start,
            self.line_end,
        ) = self.get_start_end_times_with_threshold(self.line_clock, self.k)
        (
            self.frame_start,
            self.frame_end,
        ) = self.get_start_end_times_with_threshold(self.frame_clock, self.k)

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
        signal: np.ndarray, k: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the start and end times of the on periods of the signal.
        Works for analog signals that have a squared pulse shape.

        Parameters
        ----------
        signal : np.ndarray
            An analog signal.
        k : float
            The factor used to quantify the threshold.

        Returns
        -------
        Tuple(np.ndarray, np.ndarray)
            The start and end times of the on periods of the signal.
        """

        mean = np.mean(signal)
        std = np.std(signal)
        threshold = mean + k * std

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
            idx
            for i in range(self.number_of_rotations + 1)
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
            raise ValueError("Number of rotations is not as expected")

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

        # find very short rotation periods in self.interpolated_angles

        self.config["analog_signals_processing"][
            "angle_interpolation_artifact_threshold"
        ]
        thresholded = np.zeros_like(self.interpolated_angles)
        thresholded[np.abs(self.interpolated_angles) > 0.15] = 1
        rotation_start = np.where(np.diff(thresholded) > 0)[0]
        rotation_end = np.where(np.diff(thresholded) < 0)[0]

        assert len(rotation_start) == len(rotation_end)
        assert len(rotation_start) == self.number_of_rotations

        for i, (start, end) in enumerate(
            zip(rotation_start[1:], rotation_end[:-1])
        ):
            self.interpolated_angles[end:start] = 0

        self.interpolated_angles[: rotation_start[0]] = 0
        self.interpolated_angles[rotation_end[-1] :] = 0

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
            Path(self.config["paths_write"]["debug_plots_folder"])
            / "rotation_ticks_and_rotation_on.png"
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

        plt.savefig(
            Path(self.config["paths_write"]["debug_plots_folder"])
            / "rotation_angles.png"
        )

    def rotate_frames_line_by_line(self) -> np.ndarray:
        """Rotates the image stack line by line, using the rotation angles
        by line calculated from the analog signals.

        Description of the algorithm:
        - takes one line from the image stack
        - creates a new image with only that line
        - rotates the line by the given angle
        - substitutes the line in the new image
        - adds the new image to the rotated image stack

        Edge cases and how they are handled:
        - the rotation starts in the middle of the image -> the previous lines
        are copied from the first frame
        - the rotation ends in the middle of the image -> the remaining lines
        are copied from the last frame

        Returns
        -------
        np.ndarray
            The rotated image stack.
        """
        logging.info("Starting derotation by line...")

        rotated_image_stack = rotate_an_image_array_line_by_line(
            self.image_stack,
            self.rot_deg_line,
        )

        logging.info("✨ Image stack rotated ✨")
        return rotated_image_stack

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
        for i in range(len(df)):
            row = df.loc[i]
            if np.abs(row["rotation_angle"]) > 0.0:
                adding_roatation = True
                row["direction"] = self.direction[rotation_counter]
                row["speed"] = self.speed[rotation_counter]
                row["rotation_count"] = rotation_counter

                df.loc[i] = row
            if (
                rotation_counter < 79
                and adding_roatation
                and np.abs(df.loc[i + 1, "rotation_angle"]) == 0.0
            ):
                rotation_counter += 1
                adding_roatation = False

        df.to_csv(
            self.config["paths_write"]["derotated_tiff_folder"]
            + self.config["paths_write"]["saving_name"]
            + ".csv",
            index=False,
        )
