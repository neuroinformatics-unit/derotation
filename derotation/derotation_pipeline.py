from scipy.signal import find_peaks

from derotation.analog_preprocessing import (
    apply_rotation_direction,
    check_number_of_rotations,
    find_rotation_for_each_frame_from_motor,
    get_missing_frames,
    get_starting_and_ending_frames,
    when_is_rotation_on,
)
from derotation.find_centroid import find_centroid_pipeline
from derotation.get_data import get_data


class DerotationPipeline:
    def __init__(self, path):
        (
            self.image,
            self.frame_clock,
            self.line_clock,
            self.full_rotation,
            self.rotation_ticks,
            self.dt,
            self.config,
            self.direction,
        ) = get_data(path)
        self.rot_deg = 360

        print("Data loaded")

    def process_analog_signals(self):
        self.missing_frames, self.diffs = get_missing_frames(self.frame_clock)
        (
            self.frames_start,
            self.frames_end,
            self.threshold,
        ) = get_starting_and_ending_frames(self.frame_clock, self.image)
        #  find the peaks of the rot_tick2 signal
        rotation_ticks_peaks = find_peaks(
            self.rotation_ticks,
            height=4,
            distance=20,
        )[0]

        check_number_of_rotations(
            rotation_ticks_peaks, self.direction, self.rot_deg, self.dt
        )
        self.rotation_on = when_is_rotation_on(self.full_rotation)
        self.rotation_on = apply_rotation_direction(
            self.rotation_on, self.direction
        )
        (
            self.image_rotation_degree_per_frame,
            self.signed_rotation_degrees,
        ) = find_rotation_for_each_frame_from_motor(
            self.frame_clock,
            rotation_ticks_peaks,
            self.rotation_on,
            self.frames_start,
        )

        print("Analog signals processed")

    def calculate_centers(self, img):
        lower_threshold = -2700
        higher_threshold = -2600
        binary_threshold = 32
        sigma = 2.5

        defoulting_parameters = [
            lower_threshold,
            higher_threshold,
            binary_threshold,
            sigma,
        ]

        return find_centroid_pipeline(img, defoulting_parameters)
