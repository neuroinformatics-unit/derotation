import copy

import numpy as np
import scipy.signal as signal

from derotation.analysis.derotation_pipeline import DerotationPipeline


class TestRotationSignal:
    def __init__(self):
        self.full_rotation = self.make_dummy_signal()
        self.direction = self.make_dummy_direction()
        self.k = 3
        self.min_rotation_length = 5

    def make_dummy_signal(self):
        dummy_signal = signal.square(np.linspace(0, 10, 10000))
        dummy_signal += np.random.normal(0, 0.1, 10000)

        return dummy_signal

    def make_dummy_direction(self):
        direction = np.zeros(10)
        direction[::2] = 1
        direction[1::2] = -1
        return direction

    def find_when_is_rotation_on(self):
        # old implementation
        # identify the rotation ticks that correspond to
        # clockwise and counter clockwise rotations
        threshold = 0.5  # Threshold to consider "on" or rotation occurring
        rotation_on = np.zeros_like(self.full_rotation)
        rotation_on[self.full_rotation > threshold] = 1
        return rotation_on

    def apply_rotation_direction(self, rotation_on):
        # old implementation
        rotation_signal_copy = copy.deepcopy(rotation_on)
        latest_rotation_on_end = 0

        i = 0

        rotation_blocks_idx = {"start": [], "end": []}
        while i < len(self.direction):
            # find the first rotation_on == 1
            try:
                first_rotation_on = np.where(rotation_signal_copy == 1)[0][0]
            except IndexError:
                #  no more rotations, data is over
                break
            # now assign the value in dir to all the first set of ones
            len_first_group = np.where(
                rotation_signal_copy[first_rotation_on:] == 0
            )[0][0]

            if first_rotation_on < self.min_rotation_length:
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
            rotation_on[start:end] = self.direction[i]

            latest_rotation_on_end = (
                latest_rotation_on_end + first_rotation_on + len_first_group
            )
            rotation_signal_copy = rotation_signal_copy[
                first_rotation_on + len_first_group :
            ]
            i += 1
            rotation_blocks_idx["start"].append(start)
            rotation_blocks_idx["end"].append(end)

        return rotation_on, rotation_blocks_idx

    def test_finding_correct_start_end_times_for_full_rotation(self):
        _, test_rotation_blocks_idx = self.apply_rotation_direction(
            self.find_when_is_rotation_on()
        )

        start, end = DerotationPipeline.get_start_end_times_with_threshold(
            self.full_rotation, self.k
        )
        rotation_blocks_idx = (
            DerotationPipeline.clean_start_and_end_rotation_signal(start, end)
        )

        assert len(rotation_blocks_idx["start"]) == len(
            test_rotation_blocks_idx["start"]
        )
        assert len(rotation_blocks_idx["end"]) == len(self.direction)

        assert np.all(
            rotation_blocks_idx["start"] == test_rotation_blocks_idx["start"]
        )
        assert np.all(
            rotation_blocks_idx["end"] == test_rotation_blocks_idx["end"]
        )

    def test_calculating_signed_rotation_on(self):
        test_rotation_on, _ = self.apply_rotation_direction(
            self.find_when_is_rotation_on()
        )

        start, end = DerotationPipeline.get_start_end_times_with_threshold(
            self.full_rotation, self.k
        )
        rotation_blocks_idx = (
            DerotationPipeline.clean_start_and_end_rotation_signal(start, end)
        )

        rotation_on = DerotationPipeline.create_signed_rotation_array(
            len(self.full_rotation),
            rotation_blocks_idx["start"],
            rotation_blocks_idx["end"],
            self.direction,
        )

        assert np.all(rotation_on == test_rotation_on)
