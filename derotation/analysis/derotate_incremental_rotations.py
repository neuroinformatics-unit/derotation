import copy
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
        self.number_of_rotations = self.rot_deg // self.small_rotations

    def __call__(self):
        super().process_analog_signals()
        # self.check_number_of_frame_angles()
        rotated_images = self.roatate_by_frame()
        masked_unregistered = self.add_circle_mask(rotated_images)

        mean_images = self.calculate_mean_images(masked_unregistered)
        target_image = self.get_target_image(masked_unregistered)
        shifts = self.get_shifts_using_phase_cross_correlation(
            mean_images, target_image
        )
        x_fitted, y_fitted = self.polinomial_fit(shifts)
        registered_images = self.register_rotated_images(
            masked_unregistered, x_fitted, y_fitted
        )

        masked = self.add_circle_mask(registered_images)
        self.save(masked)
        self.save_csv_with_derotation_data()

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

    def check_number_of_frame_angles(self):
        if len(self.rot_deg_frame) != self.num_frames:
            raise ValueError(
                "Number of rotation angles by frame is not equal to the "
                + "number of frames in the image stack.\n"
                + f"Number of angles: {len(self.rot_deg_frame)}\n"
                + f"Number of frames: {self.num_frames}"
            )

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

    def calculate_mean_images(self, rotated_image_stack):
        logging.info("Calculating mean images...")

        #  there is a bug in the frame start time calculation
        #  there are two additional frame starting points
        angles_subset = copy.deepcopy(self.rot_deg_frame[2:])
        # also there is a bias on the angles
        angles_subset += -0.1
        rounded_angles = np.round(angles_subset, 2)

        mean_images = []
        for i in np.arange(10, 360, 10):
            images = rotated_image_stack[rounded_angles == i]
            mean_image = np.mean(images, axis=0)

            mean_images.append(mean_image)

        #  drop the first because it is the same as the last (0, 360)
        # mean_images = mean_images[1:]

        return mean_images

    @staticmethod
    def get_target_image(rotated_image_stack):
        return np.mean(rotated_image_stack[:100], axis=0)

    def get_shifts_using_phase_cross_correlation(
        self, mean_images, target_image
    ):
        logging.info("Calculating shifts using phase cross correlation...")
        shifts = {"x": [], "y": []}
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

    def polinomial_fit(self, shifts):
        logging.info("Fitting polinomial to shifts...")
        # 0 deg shifts should be 0, insert new value
        shifts["x"].insert(0, 0)
        shifts["y"].insert(0, 0)

        angles_range = np.arange(0, 360, 10)
        x = shifts["x"]
        y = shifts["y"]

        x_fitted = np.polyfit(angles_range, x, 6)
        y_fitted = np.polyfit(angles_range, y, 6)

        return x_fitted, y_fitted

    def register_rotated_images(self, rotated_image_stack, x_fitted, y_fitted):
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
