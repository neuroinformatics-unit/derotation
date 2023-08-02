from scipy.signal import find_peaks

from derotation.analysis.analog_preprocessing import (
    apply_rotation_direction,
    check_number_of_rotations,
    find_rotation_for_each_frame_from_motor,
    find_rotation_for_each_line_from_motor,
    get_starting_and_ending_times,
    when_is_rotation_on,
)
from derotation.analysis.find_centroid import (
    find_centroid_pipeline,
    in_region,
    not_center_of_image,
)
from derotation.load_data.get_data import get_data


class DerotationPipeline:
    def __init__(self):
        (
            self.images_stack,
            self.frame_clock,
            self.line_clock,
            self.full_rotation,
            self.rotation_ticks,
            self.dt,
            self.config,
            self.direction,
        ) = get_data()
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
        check_number_of_rotations(
            self.rotation_ticks_peaks, self.direction, self.rot_deg, self.dt
        )
        self.rotation_on = when_is_rotation_on(self.full_rotation)
        self.rotation_on = apply_rotation_direction(
            self.rotation_on, self.direction
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

    def get_clean_centroids(self):
        self.correct_centers = []
        for img in self.images_stack:
            centers = self._calculate_centers()
            this_center_found = False
            for c in centers:
                if (
                    not_center_of_image(c)
                    and in_region(c)
                    and not this_center_found
                ):
                    self.correct_centers.append(c)
                    this_center_found = True
            if not this_center_found:
                self.correct_centers.append(self.correct_centers[-1])

        assert len(self.correct_centers) == len(
            self.images_stack
        ), f"len(self.correct_centers) = {len(self.correct_centers)},\
            len(self.image) = {len(self.images_stack)}"
