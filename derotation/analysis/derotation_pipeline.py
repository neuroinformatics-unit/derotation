from scipy.signal import find_peaks

from derotation.analysis.analog_preprocessing import (
    apply_rotation_direction,
    check_number_of_rotations,
    find_rotation_for_each_frame_from_motor,
    get_missing_frames,
    get_starting_and_ending_frames,
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
            self.image,
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

        return find_centroid_pipeline(self.image, defoulting_parameters)

    def get_clean_centroids(self):
        self.correct_centers = []
        for img in self.image:
            centers = self._calculate_centers(img)
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
            self.image
        ), f"len(self.correct_centers) = {len(self.correct_centers)},\
            len(self.image) = {len(self.image)}"
