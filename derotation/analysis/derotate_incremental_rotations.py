import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm

from derotation.analysis.derotation_pipeline import DerotationPipeline


class DerotateIncremental(DerotationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.small_rotations = 10
        self.number_of_rotations = len(self.rot_deg // self.small_rotations)

    def __call__(self):
        super().process_analog_signals()
        rotated_images = self.roatate_by_frame()
        masked = self.add_circle_mask(rotated_images)
        self.save(masked)

    def is_number_of_ticks_correct(self) -> bool:
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

    def roatate_by_frame(self):
        min_value_img = np.min(self.image_stack)
        new_rotated_image_stack = (
            np.ones_like(self.image_stack) * min_value_img
        )

        for idx, frame in tqdm(
            enumerate(self.image_stack), total=self.num_frames
        ):
            new_rotated_image_stack[idx] = rotate(
                frame,
                self.rot_deg_frame[idx],
                reshape=False,
                order=0,
                mode="constant",
            )
        logging.info("Finished rotating the image stack")

        return new_rotated_image_stack

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

        plt.savefig(
            Path(self.config["paths_write"]["debug_plots_folder"])
            / "rotation_angles.png"
        )
