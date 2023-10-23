import copy
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import tifffile as tiff
import tqdm
import yaml
from fancylog import fancylog
from scipy.ndimage import rotate
from scipy.signal import find_peaks
from tifffile import imsave

from derotation.load_data.custom_data_loaders import (
    get_analog_signals,
    get_rotation_direction,
)


class DerotationPipeline:
    def __init__(self, config_name):
        self.config = self.get_config(config_name)
        self.start_logging()
        self.load_data()

    def get_config(self, config_name):
        path_config = "derotation/config/" + config_name + ".yml"

        with open(Path(path_config), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        return config

    def start_logging(self):
        path = self.config["paths_write"]["logs_folder"]
        Path(path).mkdir(parents=True, exist_ok=True)
        fancylog.start_logging(
            output_dir=str(path),
            package=sys.modules[__name__.partition(".")[0]],
            filename="derotation",
            verbose=False,
        )

    def load_data(self):
        logging.info("Loading data...")

        self.image_stack = tiff.imread(
            self.config["paths_read"]["path_to_tif"]
        )

        self.num_frames = self.image_stack.shape[0]
        self.num_lines_per_frame = self.image_stack.shape[1]
        self.num_total_lines = self.num_frames * self.num_lines_per_frame

        self.direction = get_rotation_direction(
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

        self.rotation_increment = self.config["rotation_increment"]
        self.rot_deg = self.config["rot_deg"]
        self.assume_full_rotation = self.config["assume_full_rotation"]
        self.adjust_increment = self.config["adjust_increment"]
        self.rotation_kind = self.config["rotation_kind"]

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

    def process_analog_signals(self):
        self.rotation_ticks_peaks = self.find_rotation_peaks()

        start, end = self.get_start_end_times_with_threshold(
            self.full_rotation, self.k
        )
        self.rot_blocks_idx = self.correct_start_and_end_rotation_signal(
            self.inter_rotation_interval_min_len, start, end
        )
        self.rotation_on = self.create_signed_rotation_array(
            len(self.full_rotation),
            self.rot_blocks_idx["start"],
            self.rot_blocks_idx["end"],
            self.direction,
        )

        if self.debugging_plots:
            self.plot_rotation_on_and_ticks_for_inspection()

        self.check_number_of_rotations()
        if not self.is_number_of_ticks_correct() and self.adjust_increment:
            if self.assume_full_rotation:
                (
                    self.corrected_increments,
                    self.ticks_per_rotation,
                ) = self.adjust_rotation_increment(self.rotation_increment)
            else:
                self.corrected_increments = (
                    self.adjust_rotation_increment_for_incremental_changes()
                )
                logging.info(
                    f"Corrected increments: {self.corrected_increments}"
                )

        #  ===================================
        #  Quantify the rotation for each line of each frame
        (
            self.lines_start,
            self.lines_end,
        ) = self.get_starting_and_ending_times(
            clock="line", target_len=256 * len(self.image_stack)
        )
        if self.assume_full_rotation:
            (
                self.rot_deg_line,
                self.signed_rotation_degrees_line,
            ) = self.find_rotation_for_each_line_from_motor()
        else:
            self.rot_deg_line = (
                self.find_rotation_angles_by_line_in_incremental_rotation()
            )

        print("Analog signals processed")

    def find_rotation_peaks(self):
        #  scipy method works well, it's enough for our purposes

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
    def get_start_end_times_with_threshold(signal, k):
        # identify the rotation ticks that correspond to
        # clockwise and counter clockwise rotations

        mean = np.mean(signal)
        std = np.std(signal)
        threshold = mean + k * std

        thresholded_signal = np.zeros_like(signal)
        thresholded_signal[signal > threshold] = 1

        start = np.where(np.diff(thresholded_signal) > 0)[0]
        end = np.where(np.diff(thresholded_signal) < 0)[0]

        return start, end

    @staticmethod
    def correct_start_and_end_rotation_signal(
        inter_rotation_interval_min_len, start, end
    ):
        """removes very short intervals of off signal,
        which are not full rotations"""

        logging.info("Cleaning start and end rotation signal...")

        shifted_end = np.roll(end, 1)
        mask = start - shifted_end > inter_rotation_interval_min_len
        mask[0] = True  # first rotation is always a full rotation
        shifted_mask = np.roll(mask, -1)
        new_start = start[mask]
        new_end = end[shifted_mask]

        return {"start": new_start, "end": new_end}

    @staticmethod
    def create_signed_rotation_array(len_full_rotation, start, end, direction):
        logging.info("Creating signed rotation array...")
        rotation_on = np.zeros(len_full_rotation)
        for i, (start, end) in enumerate(
            zip(
                start,
                end,
            )
        ):
            rotation_on[start:end] = direction[i]

        return rotation_on

    def plot_rotation_on_and_ticks_for_inspection(self):
        #  visual inspection of the rotation ticks and the rotation on signal

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

    def check_number_of_rotations(self):
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

    def is_number_of_ticks_correct(self):
        # sanity check for the number of rotation ticks

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

    def adjust_rotation_increment_for_incremental_changes(
        self, given_increment=0.2
    ):
        total_ticks_number = len(self.rotation_ticks_peaks)

        if total_ticks_number == self.expected_tiks_per_rotation:
            return given_increment
        else:
            return self.rot_deg / total_ticks_number

    def adjust_rotation_increment(self, given_increment=0.2):
        ticks_per_rotation: int = []
        increments_per_rotation = []
        for i, (start, end) in enumerate(
            zip(
                self.rot_blocks_idx["start"],
                self.rot_blocks_idx["end"],
            )
        ):
            peaks_in_this_rotation = np.where(
                np.logical_and(
                    self.rotation_ticks_peaks > start,
                    self.rotation_ticks_peaks < end,
                )
            )[0].shape[0]
            if peaks_in_this_rotation == self.expected_tiks_per_rotation:
                increments_per_rotation.append(given_increment)
            else:
                logging.warning(
                    "Rotation {} is missing or gaining {} ticks".format(
                        i,
                        self.expected_tiks_per_rotation
                        - peaks_in_this_rotation,
                    )
                )
                increments_per_rotation.append(
                    self.rot_deg / peaks_in_this_rotation
                )
            ticks_per_rotation.append(peaks_in_this_rotation)

        return increments_per_rotation, ticks_per_rotation

    def find_rotation_angles_by_frame_in_incremental_rotation(self):
        frame_start, frame_end = self.get_start_end_times_with_threshold(
            self.frame_clock, self.k
        )

        rotation_increment_by_frame = np.zeros(len(self.image_stack))
        total_rotation_of_this_frame = 0
        for frame_id, start in enumerate(frame_start):
            try:
                ticks_in_this_frame = np.where(
                    np.logical_and(
                        self.rotation_ticks_peaks > start,
                        self.rotation_ticks_peaks < frame_start[frame_id + 1],
                    )
                )[0].shape[0]
            except IndexError:
                ticks_in_this_frame = np.where(
                    np.logical_and(
                        self.rotation_ticks_peaks > start,
                        self.rotation_ticks_peaks < frame_end[-1],
                    )
                )[0].shape[0]
            total_rotation_of_this_frame += (
                ticks_in_this_frame * self.corrected_increments
            )

            rotation_increment_by_frame[
                frame_id
            ] = total_rotation_of_this_frame

        return rotation_increment_by_frame

    def find_rotation_angles_by_line_in_incremental_rotation(self):
        #  calculate the rotation degrees for each line
        rotation_increment_by_line = np.zeros(len(self.image_stack) * 256)
        total_rotation_of_this_line = 0
        for line_id, start in enumerate(self.lines_start):
            try:
                ticks_in_this_line = np.where(
                    np.logical_and(
                        self.rotation_ticks_peaks > start,
                        self.rotation_ticks_peaks
                        < self.lines_start[line_id + 1],
                    )
                )[0].shape[0]
            except IndexError:
                ticks_in_this_line = np.where(
                    np.logical_and(
                        self.rotation_ticks_peaks > start,
                        self.rotation_ticks_peaks < self.lines_end[-1],
                    )
                )[0].shape[0]
            total_rotation_of_this_line += (
                ticks_in_this_line * self.corrected_increments
            )

            try:
                rotation_increment_by_line[
                    line_id
                ] = total_rotation_of_this_line
            except IndexError:
                break

        return rotation_increment_by_line

    def roatate_by_frame_incremental(self):
        new_rotated_image_stack = np.zeros_like(self.image_stack)

        rotation_increment_by_frame = (
            self.find_rotation_angles_by_frame_in_incremental_rotation()
        )

        for idx, frame in enumerate(self.image_stack):
            new_rotated_image_stack[idx] = rotate(
                frame,
                rotation_increment_by_frame[idx],
                reshape=False,
                order=0,
                mode="constant",
            )
            logging.info(f"Frame {idx} rotated")

        return new_rotated_image_stack

    def find_rotation_for_each_line_from_motor(self):
        #  calculate the rotation degrees for each line
        rotation_degrees = np.empty_like(self.line_clock)
        rotation_degrees[0] = 0
        rotation_increment: float = 0
        tick_peaks_corrected = np.insert(
            self.rotation_ticks_peaks, 0, 0, axis=0
        )

        for i in range(1, len(tick_peaks_corrected)):
            try:
                rotation_idx = np.where(
                    self.rot_blocks_idx["end"] > tick_peaks_corrected[i],
                )[0][0]
            except IndexError:
                logging.warning("End of rotations reached")

            if self.assume_full_rotation:
                increment = self.corrected_increments[rotation_idx]
            else:
                increment = self.rotation_increment

            time_interval = (
                tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
            )
            if (
                (time_interval > 2000)
                and (i != 0)
                and self.assume_full_rotation
            ):
                #  we cannot trust the number of ticks
                # to understand if a rotation is finished
                #  therefore we wait the rotation off signal
                rotation_increment = 0
                rotation_array = np.zeros(time_interval)
            else:
                rotation_array = np.linspace(
                    rotation_increment,
                    rotation_increment + increment,
                    time_interval,
                    endpoint=True,
                )
                rotation_increment += increment
            rotation_degrees[
                tick_peaks_corrected[i - 1] : tick_peaks_corrected[i]
            ] = rotation_array

        if self.assume_full_rotation:
            signed_rotation_degrees = rotation_degrees * self.rotation_on
        else:
            signed_rotation_degrees = rotation_degrees

        image_rotation_degree_per_line = signed_rotation_degrees[
            self.lines_start
        ]
        image_rotation_degree_per_line *= -1

        return image_rotation_degree_per_line, signed_rotation_degrees

    def rotate_frames_line_by_line(self):
        #  fill new_rotated_image_stack with non-rotated images first
        _, height, _ = self.image_stack.shape

        rotated_image_stack = copy.deepcopy(self.image_stack)
        previous_image_completed = True
        rotation_completed = True

        min_value_img = np.min(self.image_stack)

        # use tqdm
        for i, rotation in tqdm.tqdm(enumerate(self.rot_deg_line)):
            line_counter = i % height
            image_counter = i // height
            is_rotating = np.absolute(rotation) > 0.00001
            image_scanning_completed = line_counter == (height - 1)
            if not self.assume_full_rotation and i == 0:
                rotation_just_finished = False
            else:
                rotation_just_finished = not is_rotating and (
                    np.absolute(self.rot_deg_line[i - 1])
                    > np.absolute(rotation)
                )

            if is_rotating:
                if rotation_completed and (line_counter != 0):
                    #  starting a new rotation in the middle of the image
                    rotated_filled_image = (
                        np.ones_like(self.image_stack[image_counter])
                        * min_value_img
                    )  # non sampled pixels are set to the min val of the image
                    rotated_filled_image[:line_counter] = self.image_stack[
                        image_counter
                    ][:line_counter]
                elif previous_image_completed:
                    rotated_filled_image = (
                        np.ones_like(self.image_stack[image_counter])
                        * min_value_img
                    )

                rotation_completed = False

                #  we want to take the line from the row image collected
                img_with_new_lines = self.image_stack[image_counter]
                line = img_with_new_lines[line_counter]

                image_with_only_line = np.zeros_like(img_with_new_lines)
                image_with_only_line[line_counter] = line

                # is the mask really useful if we rotate with order=0?
                empty_image_mask = np.ones_like(img_with_new_lines)
                empty_image_mask[line_counter] = 0

                rotated_line = rotate(
                    image_with_only_line,
                    rotation,
                    reshape=False,
                    order=0,
                    mode="constant",
                )
                rotated_mask = rotate(
                    empty_image_mask,
                    rotation,
                    reshape=False,
                    order=0,
                    mode="constant",
                )

                #  apply rotated mask to rotated line-image
                masked = ma.masked_array(rotated_line, rotated_mask)

                #  substitute the non masked values in the new image
                rotated_filled_image = np.where(
                    masked.mask, rotated_filled_image, masked.data
                )
                previous_image_completed = False
            if (
                image_scanning_completed
                # and there_is_a_rotated_image_in_locals
                and not rotation_completed
            ) or rotation_just_finished:
                if rotation_just_finished:
                    rotation_completed = True
                    #  add missing lines at the end of the image
                    rotated_filled_image[
                        line_counter + 1 :
                    ] = self.image_stack[image_counter][line_counter + 1 :]

                # change the image in the stack inplace
                rotated_image_stack[image_counter] = rotated_filled_image
                previous_image_completed = True

                # logging.info("Image {} rotated".format(image_counter))

        return rotated_image_stack

    @staticmethod
    def add_circle_mask(rotated_image_stack):
        img_height = rotated_image_stack.shape[1]
        xx, yy = np.mgrid[:img_height, :img_height]
        circle = (xx - img_height / 2) ** 2 + (yy - img_height / 2) ** 2
        mask = circle < (img_height / 2) ** 2

        masked_img_array = []
        for img in rotated_image_stack:
            masked_img_array.append(np.where(mask, img, np.nan))

        return masked_img_array

    def save(self, masked):
        path = self.path_to_dataset_folder / "derotated"
        path.mkdir(parents=True, exist_ok=True)
        imsave(
            path / "masked_increment_with_adjustments_no_background.tif",
            np.array(masked),
        )
        logging.info(f"Masked image saved in {path}")
