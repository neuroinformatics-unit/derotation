"""
``FullPipeline`` is the main pipeline meant to be used for the full rotation
protocol, which involves rotating the sample by 360 degrees at various speeds
and directions.
"""

import copy
import itertools
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
from fancylog import fancylog
from scipy.signal import butter, find_peaks, sosfilt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tifffile import imwrite

from derotation.analysis.bayesian_optimization import BO_for_derotation
from derotation.analysis.mean_images import calculate_mean_images
from derotation.analysis.metrics import ptd_of_most_detected_blob
from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.load_data.custom_data_loaders import (
    get_analog_signals,
    read_randomized_stim_table,
)


class FullPipeline:
    """
    ``FullPipeline`` is a class that derotates an image stack
    acquired with a rotating sample under a microscope.

    It is meant to be used for the full rotation protocol, in which
    the sample is rotated by 360 degrees at various speeds and
    directions.

    It involves:
        * processing the analog signals
        * finding the offset of the image stack
        * setting the optimal center of rotation
        * derotating the image stack
        * masking the images
        * calculating the mean images
        * evaluating the quality of the derotation
        * saving the derotated image stack and the csv file

    In the constructor, it loads the config file, starts the logging
    process, and loads the data.

    Parameters
    ----------
    _config : Union[dict, str]
        Name of the config file without extension that will be retrieved
        in the derotation/config folder, or the config dictionary.
    """

    ### ----------------- Main pipeline ----------------- ###
    def __init__(self, _config: Union[dict, str]):
        """Initializes the FullPipeline class."""
        if isinstance(_config, dict):
            self.config = _config
        else:
            self.config = self.get_config(_config)

        self.start_logging()
        self.load_data()

    def __call__(self):
        """Execute the steps necessary to derotate the image stack
        from start to finish.
        """
        self.process_analog_signals()

        self.offset = self.find_image_offset(self.image_stack[0])
        self.set_optimal_center()

        rotated_images = self.derotate_frames_line_by_line()
        self.masked_image_volume = self.add_circle_mask(
            rotated_images, self.mask_diameter
        )
        self.mean_images = calculate_mean_images(
            self.masked_image_volume, self.rot_deg_frame, round_decimals=0
        )
        self.metric = ptd_of_most_detected_blob(
            self.mean_images,
            plot=self.debugging_plots,
            debug_plots_folder=self.debug_plots_folder,
        )

        self.save(self.masked_image_volume)
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

        path_config = (
            Path(__file__).parent.parent / f"config/{config_name}.yml"
        )

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
        #  suppress debug messages from matplotlib
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    def load_data(self):
        """Loads data from the paths specified in the config file.
        Data is stored in the class attributes.

        What is loaded:
            * various parameters from config file
            * image stack (tif file)
            * direction and speed of rotation (from randperm file, uses
              ``custom_data_loaders.read_randomized_stim_table``)
            * analog signals (from aux file, uses
              ``custom_data_loaders.get_analog_signals``)

        Analog signals are four files, measured in "clock_time":
            * ``frame_clock``: on during acquisition of a new frame, off
              otherwise
            * ``line_clock``: on during acquisition of a new line, off
              otherwise
            * ``full_rotation``: when the motor is rotating
            * ``rotation_ticks``: peaks at every given increment of rotation

        The data is loaded using the custom_data_loaders module, which are
        specific to the setup used in the lab. Please edit them to load
        data from your setup.
        """
        logging.info("Loading data...")
        logging.info(f"Loading {self.config['paths_read']['path_to_tif']}")
        self.image_stack = tiff.imread(
            self.config["paths_read"]["path_to_tif"]
        )

        self.num_frames = self.image_stack.shape[0]
        self.num_lines_per_frame = self.image_stack.shape[1]
        self.num_total_lines = self.num_frames * self.num_lines_per_frame
        self.mask_diameter = copy.deepcopy(self.num_lines_per_frame)

        randperm_filetype = self.config["paths_read"][
            "path_to_randperm"
        ].split(".")[-1]
        if randperm_filetype == "csv":
            table = pd.read_csv(self.config["paths_read"]["path_to_randperm"])
            self.speed = table["speed"].to_numpy()
            self.direction = table["direction"].to_numpy()
        elif randperm_filetype == "mat":
            self.direction, self.speed = read_randomized_stim_table(
                self.config["paths_read"]["path_to_randperm"]
            )
        logging.info(f"Number of rotations: {len(self.direction)}")

        rotation_direction = pd.DataFrame(
            {"direction": self.direction, "speed": self.speed}
        ).pivot_table(
            index="direction", columns="speed", aggfunc="size", fill_value=0
        )
        #  print pivot table
        logging.info(f"Rotation direction: \n{rotation_direction}")

        self.number_of_rotations = len(self.direction)

        aux_filetype = self.config["paths_read"]["path_to_aux"].split(".")[-1]
        if aux_filetype == "bin":
            (
                self.frame_clock,
                self.line_clock,
                self.full_rotation,
                self.rotation_ticks,
            ) = get_analog_signals(
                self.config["paths_read"]["path_to_aux"],
                self.config["channel_names"],
            )
        if aux_filetype == "npy":
            aux_data = np.load(self.config["paths_read"]["path_to_aux"])
            self.frame_clock = aux_data[0]
            self.line_clock = aux_data[1]
            self.full_rotation = aux_data[2]
            self.rotation_ticks = aux_data[3]

        self.total_clock_time = len(self.frame_clock)

        self.line_use_start = self.config["interpolation"]["line_use_start"]
        self.frame_use_start = self.config["interpolation"]["frame_use_start"]

        self.rotation_increment = self.config["rotation_increment"]
        self.rot_deg = self.config["rot_deg"]

        self.filename_raw = Path(
            self.config["paths_read"]["path_to_tif"]
        ).stem.split(".")[0]
        self.filename = self.config["paths_write"]["saving_name"]
        Path(self.config["paths_write"]["derotated_tiff_folder"]).mkdir(
            parents=True, exist_ok=True
        )
        self.file_saving_path_with_name = (
            Path(self.config["paths_write"]["derotated_tiff_folder"])
            / self.filename
        )
        self.std_coef = self.config["analog_signals_processing"][
            "squared_pulse_k"
        ]
        self.inter_rotation_interval_min_len = self.config[
            "analog_signals_processing"
        ]["inter_rotation_interval_min_len"]

        self.debugging_plots = self.config["debugging_plots"]

        self.frame_rate = self.config["frame_rate"]

        if self.debugging_plots:
            self.debug_plots_folder = Path(
                self.config["paths_write"]["debug_plots_folder"]
            )
            Path(self.debug_plots_folder).mkdir(parents=True, exist_ok=True)

            #  unlink previous debug plots
            logging.info("Deleting previous debug plots...")
            for item in self.debug_plots_folder.iterdir():
                if item.is_dir():
                    for file in item.iterdir():
                        if file.suffix == ".png":
                            file.unlink()
                else:
                    if item.suffix == ".png":
                        item.unlink()

        logging.info(f"Dataset {self.filename_raw} loaded")
        logging.info(f"Filename: {self.filename}")

        #  by default the center of rotation is the center of the image
        if not self.config["biased_center"]:
            self.center_of_rotation = (
                self.num_lines_per_frame // 2,
                self.num_lines_per_frame // 2,
            )
        else:
            self.center_of_rotation = tuple(self.config["biased_center"])

        self.hooks = {}
        self.rotation_plane_angle = 0
        self.rotation_plane_orientation = 0

        self.delta = self.config["delta_center"]
        self.init_points = self.config["init_points"]
        self.n_iter = self.config["n_iter"]

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

        If debugging_plots is ``True``, it also plots:
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
            self.line_clock,
            self.std_coef,
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
            self.plot_rotation_angles_and_velocity()
            self.plot_rotation_speeds()

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

        Used the ``inter_rotation_interval_min_len`` parameter from the config
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
            logging.warning(
                f"Number of rotations is not {self.number_of_rotations}."
                + f"Found {self.rot_blocks_idx['start'].shape[0]} rotations."
                + "Adjusting starting and ending times..."
            )
            self.find_missing_rotation_on_periods()

        logging.info("Number of rotations is as expected")

    def find_missing_rotation_on_periods(self):
        """
        Find the missing rotation on periods by looking at the rotation ticks
        and the rotation on signal. This is useful when the number of rotations
        is not as expected.
        Uses k-means to cluster the ticks and pick the first and last for each
        cluster. These are the starting and ending times.
        """

        kmeans = KMeans(
            n_clusters=self.number_of_rotations, random_state=0
        ).fit(self.rotation_ticks_peaks.reshape(-1, 1))

        new_start = np.zeros(self.number_of_rotations, dtype=int)
        new_end = np.zeros(self.number_of_rotations, dtype=int)
        for i in range(self.number_of_rotations):
            new_start[i] = self.rotation_ticks_peaks[kmeans.labels_ == i].min()
            new_end[i] = self.rotation_ticks_peaks[kmeans.labels_ == i].max()

        #  cluster number is not the same as the number of rotations
        self.rot_blocks_idx["start"] = sorted(new_start)
        self.rot_blocks_idx["end"] = sorted(new_end)

        #  update the rotation on signal
        self.rotation_on = self.create_signed_rotation_array()

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
        thresholded[np.abs(self.interpolated_angles) > 0.2] = 1
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
            #  plot the rotation on signal and the interpolated angles
            #  this is useful to debug the interpolation
            self.plot_rotation_on_and_ticks()

            logging.warning(
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

    def calculate_velocity(self):
        """Calculates the velocity of the rotation by line."""
        # Compute the correct sampling rate
        self.sampling_rate = (
            self.frame_rate * self.num_lines_per_frame
        )  # 1725.44 Hz

        # Unwrap angles and compute velocity
        warr = np.rad2deg(np.unwrap(np.deg2rad(self.rot_deg_line)))
        velocity = np.diff(warr) * self.sampling_rate

        # Butterworth low-pass filter
        order = 3
        nyq = 0.5 * self.sampling_rate  # Nyquist frequency
        cutoff = 10 / nyq  # Normalized cutoff frequency

        sos = butter(
            order, cutoff, btype="low", output="sos"
        )  # Use 'sos' for stability
        filtered = sosfilt(sos, velocity)  # Apply filter correctly

        return filtered

    def plot_rotation_on_and_ticks(self):
        """Plots the rotation ticks and the rotation on signal.
        This plot will be saved in the ``debug_plots`` folder.
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
        plt.close()

    def plot_rotation_angles_and_velocity(self):
        """Plots example rotation angles by line and frame for each speed.
        The velocity is also plotted on top of the rotation angles.

        This plot will be saved in the ``debug_plots`` folder. Please inspect
        it to check that the rotation angles are correctly calculated.
        """
        logging.info("Plotting rotation angles...")

        speeds = set(self.speed)
        logging.info(f"Speeds: {speeds}")
        logging.info(f"len (speeds): {len(speeds)}")
        first_idx_for_each_speed = [
            np.where(self.speed == s)[0][0] for s in speeds
        ]
        first_idx_for_each_speed = sorted(first_idx_for_each_speed)

        n = len(speeds)
        if n <= 2:
            n_rows, n_cols = 1, n
        else:
            n_rows, n_cols = 2, (n + 1) // 2
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(15, 7),
        )

        velocity = self.calculate_velocity()

        for i, id in enumerate(first_idx_for_each_speed):
            col = i // 2
            row = i % 2

            if isinstance(axs, np.ndarray):
                if axs.ndim == 2:
                    ax = axs[col, row]
                elif axs.ndim == 1:
                    ax = axs[row]
            else:
                ax = axs

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

            #  plot velocity on top in red
            ax2 = ax.twinx()
            ax2.plot(
                self.line_start[
                    start_line_idx:end_line_idx
                ],  # Align x-axis with line_start
                velocity[start_line_idx:end_line_idx] * -1,
                color="gray",
                label="velocity",
            )

            # remove top axis
            ax2.spines["top"].set_visible(False)

            #  set x label
            ax.set_xlabel("Time (s)")

            #  set y label left
            ax.set_ylabel("Rotation angle (°)", color="black")

            #  set y label right
            ax2.set_ylabel("Velocity (°/s)", color="gray")

        fig.suptitle("Rotation angles by line and frame")

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        plt.savefig(self.debug_plots_folder / "rotation_angles.png")
        plt.close()

    def plot_rotation_speeds(self):
        """Plots the velocity of the rotation for each speed.
        This plot will be saved in the ``debug_plots`` folder.
        Please inspect it to check that the velocity is correctly calculated.
        """

        unique_speeds = sorted(set(self.speed))
        unique_directions = sorted(set(self.direction))
        fig, axs = plt.subplots(
            len(unique_speeds), len(unique_directions), figsize=(15, 7)
        )
        velocity = self.calculate_velocity()

        #  row clockwise, column speed
        for i, (direction, speed) in enumerate(
            itertools.product(unique_directions, unique_speeds)
        ):
            row = i // len(unique_speeds)
            col = i % len(unique_speeds)
            idx_this_speed = np.where(
                np.logical_and(
                    self.speed == speed, self.direction == direction
                )
            )[0]

            #  linspace of colors depending on repetition number
            colors = plt.cm.viridis(np.linspace(0, 1, len(idx_this_speed)))

            for j, idx in enumerate(idx_this_speed):
                this_velocity = velocity[
                    self.clock_to_latest_line_start(
                        self.rot_blocks_idx["start"][idx]
                    ) : self.clock_to_latest_line_start(
                        self.rot_blocks_idx["end"][idx]
                    )
                ]
                #  if axis is two dim
                if isinstance(axs, np.ndarray):
                    if axs.ndim == 2:
                        ax = axs[col, row]
                    elif axs.ndim == 1:
                        ax = axs[row]
                else:
                    ax = axs

                ax.plot(
                    np.linspace(
                        0,
                        len(this_velocity) * self.sampling_rate,
                        len(this_velocity),
                    ),
                    this_velocity,
                    label=f"repetition {idx}",
                    color=colors[j],
                )
            ax.set_title(
                f"Speed: {speed}, direction:"
                f"{'CW' if direction == 1 else 'CCW'}"
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            #  set titles of axis
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity (°/s)")

            #  leave more space between subplots
            plt.subplots_adjust(hspace=0.5, wspace=0.5)

        fig.suptitle("Rotation on signal for each speed")
        plt.savefig(self.debug_plots_folder / "all_speeds.png")
        plt.close()

    ### ----------------- Derotation ----------------- ###

    def find_optimal_parameters(self):
        """
        Finds the optimal parameters for the derotation.
        It calls the Bayesian Optimization algorithm implemented in
        ``BO_for_derotation``.
        """
        logging.info("Finding optimal parameters...")

        bo = BO_for_derotation(
            self.image_stack,
            self.rot_deg_line,
            self.rot_deg_frame,
            self.offset,
            self.center_of_rotation,
            self.delta,
            self.init_points,
            self.n_iter,
            self.debug_plots_folder,
        )

        maximum = bo.optimize()

        logging.info(f"Optimal parameters: {maximum}")
        logging.info(f"Target: {maximum['target']}")

        if maximum["target"] > -50:
            # Consider a value of -50 as a threshold for the quality of the fit
            logging.info("Using fitted center of rotation...")
            x_center, y_center = maximum["params"].values()
            self.center_of_rotation = (x_center, y_center)

            #  write optimal center in a text file
            with open(
                self.debug_plots_folder / "optimal_center_of_rotation.txt", "w"
            ) as f:
                f.write(f"Optimal center of rotation: {x_center}, {y_center}")

    def set_optimal_center(self):
        """Checks if the optimal center of rotation is calculated.
        If it is not calculated, it will calculate it.
        """
        try:
            with open(
                self.debug_plots_folder / "optimal_center_of_rotation.txt", "r"
            ) as f:
                optimal_center = f.read()
                self.center_of_rotation = tuple(
                    map(float, optimal_center.split(":")[1].split(","))
                )
                logging.info("Optimal center of rotation read from file.")
        except FileNotFoundError:
            logging.info("Optimal center of rotation not found, calculating.")
            self.find_optimal_parameters()

    def plot_max_projection_with_center(
        self, stack, name="max_projection_with_center"
    ):
        """Plots the maximum projection of the image stack with the center
        of rotation.
        This plot will be saved in the ``debug_plots`` folder.
        Please inspect it to check that the center of rotation is correctly
        placed.
        """
        logging.info("Plotting max projection with center...")

        max_projection = np.max(stack, axis=0)

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

        plt.savefig(str(self.debug_plots_folder / name) + ".png")
        plt.close()

    def derotate_frames_line_by_line(self) -> np.ndarray:
        """Wrapper for the function ``derotate_an_image_array_line_by_line``.
        Before calling the function, it finds the F0 image offset with
        ``find_image_offset``.

        Returns
        -------
        np.ndarray
            The rotated image stack.
        """
        logging.info("Starting derotation by line...")

        if self.debugging_plots:
            self.plot_max_projection_with_center(self.image_stack)

        #  By default rotation_plane_angle and rotation_plane_orientation are 0
        #  they have to be overwritten before calling the function.
        #  To calculate them please use the ellipse fit.
        derotated_image_stack = derotate_an_image_array_line_by_line(
            self.image_stack,
            self.rot_deg_line,
            blank_pixels_value=self.offset,
            center=self.center_of_rotation,
            plotting_hook_line_addition=self.hooks.get(
                "plotting_hook_line_addition"
            ),
            plotting_hook_image_completed=self.hooks.get(
                "plotting_hook_image_completed"
            ),
            use_homography=self.rotation_plane_angle != 0,
            rotation_plane_angle=self.rotation_plane_angle,
            rotation_plane_orientation=self.rotation_plane_orientation,
        )

        if self.debugging_plots:
            self.plot_max_projection_with_center(
                derotated_image_stack,
                name="derotated_max_projection_with_center",
            )
            self.mean_image_for_each_rotation(derotated_image_stack)

        logging.info("✨ Image stack derotated ✨")
        return derotated_image_stack

    @staticmethod
    def find_image_offset(img):
        """Find the "F0", also called "image offset" for a given image.

        Explanations: What is the image offset?
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

    def mean_image_for_each_rotation(self, derotated_image_stack):
        """Calculates the mean image for each rotation and saves it in the
        ``debug_plots`` folder.
        This plot will be saved in the ``debug_plots`` folder.
        Please inspect it to check that the mean images are correctly
        calculated.
        """
        folder = self.debug_plots_folder / "mean_images"
        Path(folder).mkdir(parents=True, exist_ok=True)
        for i, (start, end) in enumerate(
            zip(self.rot_blocks_idx["start"], self.rot_blocks_idx["end"])
        ):
            frame_start = self.clock_to_latest_frame_start(start)
            frame_end = self.clock_to_latest_frame_start(end)
            mean_image = np.mean(
                derotated_image_stack[frame_start:frame_end], axis=0
            )
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(mean_image, cmap="viridis")
            ax.axis("off")
            plt.savefig(str(folder / f"mean_image_rotation_{i}.png"))
            plt.close()

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
        imwrite(
            str(self.file_saving_path_with_name) + ".tif",
            np.array(masked),
        )
        logging.info(f"Saving {str(self.file_saving_path_with_name) + '.tif'}")

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
        if len(self.rot_deg_frame) > self.num_frames:
            df["rotation_angle"] = self.rot_deg_frame[: self.num_frames]
            df["clock"] = self.frame_start[: self.num_frames]
            logging.warning(
                "Number of rotation angles by frame is higher than the"
                " number of frames"
            )
        elif len(self.rot_deg_frame) < self.num_frames:
            missing_frames = self.num_frames - len(self.rot_deg_frame)
            df["rotation_angle"] = np.append(
                self.rot_deg_frame, [0] * missing_frames
            )
            df["clock"] = np.append(self.frame_start, [0] * missing_frames)

            logging.warning(
                "Number of rotation angles by frame is lower than the"
                " number of frames. Adjusted."
            )
        else:
            df["rotation_angle"] = self.rot_deg_frame
            df["clock"] = self.frame_start

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

        df.to_csv(
            str(self.file_saving_path_with_name) + ".csv",
            index=False,
        )

        self.derotation_output_table = df
