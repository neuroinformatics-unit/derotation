import copy

import numpy as np

from derotation.analysis.opt_frame_number import find_best_k


def get_missing_frames(frame_clock):
    diffs = np.diff(frame_clock)
    missing_frames = np.where(diffs > 0.1)[0]

    return missing_frames, diffs


def get_starting_and_ending_times(clock, image, clock_type):
    # Calculate the threshold using a percentile of the total signal
    best_k = find_best_k(clock, image, clock_type=clock_type)
    threshold = np.mean(clock) + best_k * np.std(clock)
    print(f"Best threshold: {threshold}")
    start = np.where(np.diff(clock) > threshold)[0]
    end = np.where(np.diff(clock) < -threshold)[0]

    return start, end, threshold


def check_number_of_rotations(
    rotation_ticks_peaks, rotation_blocks_idx, rot_deg, given_increment=0.2
):
    print(f"Current increment: {given_increment}")
    # sanity check for the number of rotation ticks
    number_of_rotations = len(rotation_blocks_idx["start"])

    expected_tiks_per_rotation = rot_deg / given_increment
    found_ticks = len(rotation_ticks_peaks)
    expected_ticks = expected_tiks_per_rotation * number_of_rotations

    delta = len(rotation_ticks_peaks) - expected_ticks

    if expected_ticks == found_ticks:
        print(f"Number of ticks is as expected: {found_ticks}")
        return np.ones(number_of_rotations) * given_increment
    else:
        print(f"Number of ticks is not as expected: {found_ticks}")
        print(f"Expected ticks: {expected_ticks}")
        print(f"Delta: {delta}")

    corrected_increments = adjust_rotation_increment(
        rotation_ticks_peaks,
        rotation_blocks_idx,
        expected_tiks_per_rotation,
        rot_deg,
        given_increment,
    )

    return corrected_increments


def adjust_rotation_increment(
    rotation_ticks,
    rotation_blocks_idx,
    expected_tiks_per_rotation,
    rot_deg,
    given_increment=0.2,
):
    increments_per_rotation = []
    for i, (start, end) in enumerate(
        zip(rotation_blocks_idx["start"], rotation_blocks_idx["end"])
    ):
        peaks_in_this_rotation = np.where(
            np.logical_and(rotation_ticks > start, rotation_ticks < end)
        )[0].shape[0]
        if peaks_in_this_rotation == expected_tiks_per_rotation:
            increments_per_rotation.append(given_increment)
        else:
            print(
                "Rotation {} is missing or gaining {} ticks".format(
                    i, expected_tiks_per_rotation - peaks_in_this_rotation
                )
            )
            increments_per_rotation.append(rot_deg / peaks_in_this_rotation)

    return increments_per_rotation


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

    rotation_blocks_idx = {"start": [], "end": []}
    while i < len(direction):
        # find the first rotation_on == 1
        try:
            first_rotation_on = np.where(rotation_signal_copy == 1)[0][0]
        except IndexError:
            #  no more rotations, data is over
            print(f"No more rotations, missing {len(direction) - i} rotations")
            break
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

        start = latest_rotation_on_end + first_rotation_on
        end = latest_rotation_on_end + first_rotation_on + len_first_group
        rotation_on[start:end] = direction[i]
        latest_rotation_on_end = (
            latest_rotation_on_end + first_rotation_on + len_first_group
        )
        rotation_signal_copy = rotation_signal_copy[
            first_rotation_on + len_first_group :
        ]
        i += 1  # Increment the loop counter

        rotation_blocks_idx["start"].append(start)
        rotation_blocks_idx["end"].append(end)

    return rotation_on, rotation_blocks_idx


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


def find_rotation_for_each_line_from_motor(
    line_clock,
    rotation_ticks_peaks,
    rotation_on,
    lines_start,
    corrected_increments,
    rotation_blocks_idx,
):
    #  calculate the rotation degrees for each line
    rotation_degrees = np.empty_like(line_clock)
    rotation_degrees[0] = 0
    rotation_increment: float = 0
    tick_peaks_corrected = np.insert(rotation_ticks_peaks, 0, 0, axis=0)
    for i in range(1, len(tick_peaks_corrected)):
        rotation_idx = np.where(
            rotation_blocks_idx["end"] > tick_peaks_corrected[i],
        )[0][0]

        increment = corrected_increments[rotation_idx]

        time_interval = tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
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
    signed_rotation_degrees = rotation_degrees * rotation_on
    image_rotation_degree_per_line = signed_rotation_degrees[lines_start]
    image_rotation_degree_per_line *= -1

    return image_rotation_degree_per_line, signed_rotation_degrees
