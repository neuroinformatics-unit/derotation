import copy

import numpy as np
from optimizers import find_best_k


def get_missing_frames(frame_clock):
    diffs = np.diff(frame_clock)
    missing_frames = np.where(diffs > 0.1)[0]

    return missing_frames, diffs


def get_starting_and_ending_frames(frame_clock, image):
    # Calculate the threshold using a percentile of the total signal
    best_k = find_best_k(frame_clock, image)
    threshold = np.mean(frame_clock) + best_k * np.std(frame_clock)
    print(f"Best threshold: {threshold}")
    frames_start = np.where(np.diff(frame_clock) > threshold)[0]
    frames_end = np.where(np.diff(frame_clock) < -threshold)[0]

    return frames_start, frames_end, threshold


def check_number_of_rotations(rotation_ticks_peaks, direction, rot_deg, dt):
    # sanity check for the number of rotation ticks
    number_of_rotations = len(direction)
    expected_tiks_per_rotation = rot_deg / dt
    ratio = len(rotation_ticks_peaks) / expected_tiks_per_rotation
    if ratio > number_of_rotations:
        print(
            f"There are more rotation ticks than expected, \
                {len(rotation_ticks_peaks)}"
        )
    elif ratio < number_of_rotations:
        print(
            f"There are less rotation ticks than expected, \
                {len(rotation_ticks_peaks)}"
        )


def when_is_rotation_on(full_rotation):
    # identify the rotation ticks that correspond to
    # clockwise and counter clockwise rotations
    threshold = 0.5  # Threshold to consider "on" or rotation occurring
    rotation_on = np.zeros_like(full_rotation)
    rotation_on[full_rotation > threshold] = 1

    return rotation_on


def apply_rotation_direction(rotation_on, direction):
    rotation_signal_copy = copy.deepcopy(rotation_on)
    latest_rotation_on_end = 0

    i = 0
    while i < len(direction):
        # find the first rotation_on == 1
        first_rotation_on = np.where(rotation_signal_copy == 1)[0][0]
        # now assign the value in dir to all the first set of ones
        len_first_group = np.where(
            rotation_signal_copy[first_rotation_on:] == 0
        )[0][0]
        if len_first_group < 1000:
            #  skip this short rotation because it is a false one
            #  done one additional time to clean up the trace at the end
            rotation_signal_copy = rotation_signal_copy[
                first_rotation_on + len_first_group :
            ]
            latest_rotation_on_end = (
                latest_rotation_on_end + first_rotation_on + len_first_group
            )
            continue

        rotation_on[
            latest_rotation_on_end
            + first_rotation_on : latest_rotation_on_end
            + first_rotation_on
            + len_first_group
        ] = direction[i]
        latest_rotation_on_end = (
            latest_rotation_on_end + first_rotation_on + len_first_group
        )
        rotation_signal_copy = rotation_signal_copy[
            first_rotation_on + len_first_group :
        ]
        i += 1  # Increment the loop counter

    return rotation_on


def find_rotation_for_each_frame_from_motor(
    frame_clock, rotation_ticks_peaks, rotation_on, frames_start
):
    #  calculate the rotation degrees for each frame
    rotation_degrees = np.empty_like(frame_clock)
    rotation_degrees[0] = 0
    current_rotation: float = 0
    tick_peaks_corrected = np.insert(rotation_ticks_peaks, 0, 0, axis=0)
    for i in range(0, len(tick_peaks_corrected)):
        time_interval = tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
        if time_interval > 2000 and i != 0:
            current_rotation = 0
        else:
            current_rotation += 0.2
        rotation_degrees[
            tick_peaks_corrected[i - 1] : tick_peaks_corrected[i]
        ] = current_rotation
    signed_rotation_degrees = rotation_degrees * rotation_on
    image_rotation_degree_per_frame = signed_rotation_degrees[frames_start]
    image_rotation_degree_per_frame *= -1

    return image_rotation_degree_per_frame, signed_rotation_degrees
