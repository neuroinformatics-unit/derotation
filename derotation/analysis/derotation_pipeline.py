from scipy.signal import find_peaks

from derotation.analysis.analog_preprocessing import (
    apply_rotation_direction,
    check_number_of_rotations,
    find_rotation_for_each_frame_from_motor,
    find_rotation_for_each_line_from_motor,
    get_starting_and_ending_times,
    when_is_rotation_on,
)
from derotation.analysis.archive.find_centroid import (
    find_centroid_pipeline,
)
from derotation.load_data.get_data import get_data


class DerotationPipeline:
    def __init__(self, dataset_name="grid"):
        (
            self.images_stack,
            self.frame_clock,
            self.line_clock,
            self.full_rotation,
            self.rotation_ticks,
            self.dt,
            self.config,
            self.direction,
        ) = get_data(dataset_name)
        self.rot_deg = 360

        print("Data loaded")

    def process_analog_signals(self):
        #  Rotation bit of the analysis
        #  find the peaks of the rot_tick2 signal
        self.rotation_ticks_peaks = find_peaks(
            self.rotation_ticks,
            height=4,
            distance=20,
        )[0]
        self.rotation_on = when_is_rotation_on(self.full_rotation)

        self.rotation_on, self.rotation_blocks_idx = apply_rotation_direction(
            self.rotation_on, self.direction
        )

        self.corrected_increments = check_number_of_rotations(
            self.rotation_ticks_peaks,
            self.rotation_blocks_idx,
            self.rot_deg,
        )

        (
            self.frames_start,
            self.frames_end,
            self.threshold,
        ) = get_starting_and_ending_times(
            self.frame_clock, self.images_stack, clock_type="frame"
        )
        (
            self.lines_start,
            self.lines_end,
            self.threshold,
        ) = get_starting_and_ending_times(
            self.line_clock, self.images_stack, clock_type="line"
        )
        (
            self.image_rotation_degrees_line,
            self.signed_rotation_degrees_line,
        ) = find_rotation_for_each_line_from_motor(
            self.line_clock,
            self.rotation_ticks_peaks,
            self.rotation_on,
            self.lines_start,
            self.corrected_increments,
            self.rotation_blocks_idx,
        )
        (
            self.image_rotation_degrees_frame,
            self.signed_rotation_degrees_frane,
        ) = find_rotation_for_each_frame_from_motor(
            self.frame_clock,
            self.rotation_ticks_peaks,
            self.rotation_on,
            self.frames_start,
        )

        print("Analog signals processed")

    def _calculate_centers(self):
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

        return find_centroid_pipeline(self.images_stack, defoulting_parameters)
