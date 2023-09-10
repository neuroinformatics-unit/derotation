import copy
import logging
import sys
from pathlib import Path

import numpy as np
import numpy.ma as ma
from fancylog import fancylog
from scipy.ndimage import rotate
from scipy.optimize import bisect
from scipy.signal import find_peaks
from tifffile import imsave

from derotation.load_data.get_data import get_data


class DerotationPipeline:
    def __init__(self):
        Path("logs").mkdir(parents=True, exist_ok=True)
        fancylog.start_logging(
            output_dir=str(Path("logs")),
            package=sys.modules[__name__.partition(".")[0]],
            filename="derotation",
            verbose=False,
        )
        (
            self.image_stack,
            self.frame_clock,
            self.line_clock,
            self.full_rotation,
            self.rotation_ticks,
            self.rotation_increment,
            self.config,
            self.direction,
            self.path_to_dataset_folder,
        ) = get_data()
        self.rot_deg = 360

        logging.info(
            f"Dataset {self.config['paths']['dataset-folder']} loaded"
        )

    def process_analog_signals(self):
        #  ===================================
        #  Use rotation ticks to find the correct rotation angles
        self.rotation_ticks_peaks = find_peaks(
            self.rotation_ticks,
            height=4,
            distance=20,
        )[0]
        self.rotation_on = self.find_when_is_rotation_on()
        self.rot_blocks_idx = self.apply_rotation_direction()
        self.expected_tiks_per_rotation = self.check_number_of_rotations(
            self.rotation_increment
        )
        self.corrected_increments = self.adjust_rotation_increment(
            self.rotation_increment
        )

        #  ===================================
        #  Quantify the rotation for each line of each frame
        (
            self.lines_start,
            self.lines_end,
            self.threshold,
        ) = self.get_starting_and_ending_times(clock_type="line")
        (
            self.rot_deg_line,
            self.signed_rotation_degrees_line,
        ) = self.find_rotation_for_each_line_from_motor()

        print("Analog signals processed")

    def _find_rotation_peaks(self):
        # to be improved
        expected_tiks_per_rotation = self.rot_deg / self.rotation_increment

        best_k = bisect(
            self.goodness_of_threshold,
            0,
            np.max(self.rotation_ticks),
            args=(
                self.rotation_ticks,  # clock
                expected_tiks_per_rotation,  # target len
            ),
        )
        threshold = np.mean(self.rotation_ticks) + best_k * np.std(
            self.rotation_ticks
        )

        start = np.where(np.diff(self.rotation_ticks) > threshold)[0]

        return start

    def find_when_is_rotation_on(self):
        # identify the rotation ticks that correspond to
        # clockwise and counter clockwise rotations
        threshold = 0.5  # Threshold to consider "on" or rotation occurring
        rotation_on = np.zeros_like(self.full_rotation)
        rotation_on[self.full_rotation > threshold] = 1
        return rotation_on

    def apply_rotation_direction(self):
        rotation_signal_copy = copy.deepcopy(self.rotation_on)
        latest_rotation_on_end = 0

        i = 0

        rotation_blocks_idx = {"start": [], "end": []}
        while i < len(self.direction):
            # find the first rotation_on == 1
            try:
                first_rotation_on = np.where(rotation_signal_copy == 1)[0][0]
            except IndexError:
                #  no more rotations, data is over
                logging.info(
                    f"Completed, missing {len(self.direction) - i} rotations"
                )
                break
            # now assign the value in dir to all the first set of ones
            len_first_group = np.where(
                rotation_signal_copy[first_rotation_on:] == 0
            )[0][0]

            if first_rotation_on < 1000:
                #  skip this short rotation because it is a false one
                #  done one additional time to clean up the trace at the end
                rotation_signal_copy = rotation_signal_copy[
                    first_rotation_on + len_first_group :
                ]
                latest_rotation_on_end = (
                    latest_rotation_on_end
                    + first_rotation_on
                    + len_first_group
                )
                continue

            start = latest_rotation_on_end + first_rotation_on
            end = latest_rotation_on_end + first_rotation_on + len_first_group

            # rotation on is modified in place
            self.rotation_on[start:end] = self.direction[i]

            latest_rotation_on_end = (
                latest_rotation_on_end + first_rotation_on + len_first_group
            )
            rotation_signal_copy = rotation_signal_copy[
                first_rotation_on + len_first_group :
            ]
            i += 1
            rotation_blocks_idx["start"].append(start)
            rotation_blocks_idx["end"].append(end)

        return rotation_blocks_idx

    def check_number_of_rotations(self, given_increment=0.2):
        logging.info(f"Current increment: {given_increment}")
        # sanity check for the number of rotation ticks
        number_of_rotations = len(self.rot_blocks_idx["start"])

        expected_tiks_per_rotation = self.rot_deg / given_increment
        found_ticks = len(self.rotation_ticks_peaks)
        expected_ticks = expected_tiks_per_rotation * number_of_rotations

        delta = len(self.rotation_ticks_peaks) - expected_ticks

        if expected_ticks == found_ticks:
            logging.info(f"Number of ticks is as expected: {found_ticks}")
            return np.ones(number_of_rotations) * given_increment
        else:
            logging.warning(
                f"Number of ticks is not as expected: {found_ticks}"
            )
            logging.warning(f"Expected ticks: {expected_ticks}")
            logging.warning(f"Delta: {delta}")

        return expected_tiks_per_rotation

    def adjust_rotation_increment(self, given_increment=0.2):
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

        return increments_per_rotation

    def get_starting_and_ending_times(self, clock_type):
        clock = self.line_clock if clock_type == "line" else self.frame_clock
        # Calculate the threshold using a percentile of the total signal
        target_len = (
            len(self.image_stack)
            if clock_type == "frame"
            else len(self.image_stack) * 256
        )
        best_k = bisect(
            self.goodness_of_threshold, -5, 5, args=(clock, target_len)
        )
        threshold = np.mean(clock) + best_k * np.std(clock)

        try:
            start = np.where(np.diff(clock) > threshold)[0]
            assert len(start) == target_len, f"{len(start)} != {target_len}"
            logging.info(f"Best threshold: {threshold}")
        except AssertionError:
            logging.warning(
                "Suboptimal threshold found, missing "
                + f"{len(start) - target_len} line clock ticks"
            )

        start = np.where(np.diff(clock) > threshold)[0]
        end = np.where(np.diff(clock) < -threshold)[0]

        return start, end, threshold

    @staticmethod
    def goodness_of_threshold(k, clock, target_len):
        # Calculate the threshold using a percentile of the total signal
        mean = np.mean(clock)
        std = np.std(clock)
        threshold = mean + k * std

        start = np.where(np.diff(clock) > threshold)[0]
        return len(start) - target_len

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

            increment = self.corrected_increments[rotation_idx]

            time_interval = (
                tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
            )
            if time_interval > 2000 and i != 0:
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
        signed_rotation_degrees = rotation_degrees * self.rotation_on
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
        for i, rotation in enumerate(self.rot_deg_line):
            line_counter = i % height
            image_counter = i // height
            is_rotating = np.absolute(rotation) > 0.00001
            image_scanning_completed = line_counter == (height - 1)
            rotation_just_finished = not is_rotating and (
                np.absolute(self.rot_deg_line[i - 1]) > np.absolute(rotation)
            )

            if is_rotating:
                if rotation_completed and (line_counter != 0):
                    #  starting a new rotation in the middle of the image
                    rotated_filled_image = self.image_stack[image_counter]
                elif previous_image_completed:
                    # rotation in progress and new image to be rotated
                    # rotated_filled_image = np.zeros_like(
                    #     image_stack[image_counter]
                    # )
                    rotated_filled_image = self.image_stack[image_counter]

                rotation_completed = False

                #  we want to take the line from the row image collected
                img_with_new_lines = self.image_stack[image_counter]
                line = img_with_new_lines[line_counter]

                image_with_only_line = np.zeros_like(img_with_new_lines)
                image_with_only_line[line_counter] = line

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

                logging.info("Image {} rotated".format(image_counter))

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
        path = self.path_to_dataset_folder / "dertotated"
        path.mkdir(parents=True, exist_ok=True)
        imsave(
            path / "masked.tif",
            np.array(masked),
        )
        logging.info(f"Masked image saved in {path}")
