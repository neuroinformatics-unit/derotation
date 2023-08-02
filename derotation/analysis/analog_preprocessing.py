import copy

import numpy as np

from derotation.analysis.opt_frame_number import find_best_k


def get_missing_frames(frame_clock):
    diffs = np.diff(frame_clock)
    missing_frames = np.where(diffs > 0.1)[0]

    return missing_frames, diffs


def get_starting_and_ending_times(clock, image):
    # Calculate the threshold using a percentile of the total signal
    best_k = find_best_k(clock, image, clock_type="line")
    threshold = np.mean(clock) + best_k * np.std(clock)
    print(f"Best threshold: {threshold}")
    start = np.where(np.diff(clock) > threshold)[0]
    end = np.where(np.diff(clock) < -threshold)[0]

    return start, end, threshold


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


def interpolate_motor_ticks(
    rotation_ticks_peaks,
    rotation_on,
    clock,
    start,
    time_threshold,
    rotation_increment,
):
    rotation_degrees = np.empty_like(clock)
    rotation_degrees[0] = 0
    current_rotation: float = 0
    tick_peaks_corrected = np.insert(rotation_ticks_peaks, 0, 0, axis=0)

    for i in range(0, len(tick_peaks_corrected)):
        time_interval = tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
        if time_interval > time_threshold and i != 0:
            current_rotation = 0
        else:
            current_rotation += rotation_increment
        rotation_degrees[
            tick_peaks_corrected[i - 1] : tick_peaks_corrected[i]
        ] = current_rotation

    signed_rotation_degrees = rotation_degrees * rotation_on

    # rotation degrees per frame or per line
    image_rotation_degree = signed_rotation_degrees[start]
    image_rotation_degree *= -1

    return image_rotation_degree, signed_rotation_degrees


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
    line_clock, rotation_ticks_peaks, rotation_on, lines_start
):
    #  calculate the rotation degrees for each line
    rotation_degrees = np.empty_like(line_clock)
    rotation_degrees[0] = 0
    rotation_increment: float = 0
    tick_peaks_corrected = np.insert(rotation_ticks_peaks, 0, 0, axis=0)
    for i in range(1, len(tick_peaks_corrected)):
        time_interval = tick_peaks_corrected[i] - tick_peaks_corrected[i - 1]
        if time_interval > 2000 and i != 0:
            rotation_increment = 0
            rotation_array = np.zeros(time_interval)
        else:
            rotation_array = np.linspace(
                rotation_increment,
                rotation_increment + 0.2,
                time_interval,
                endpoint=True,
            )
            rotation_increment += 0.2
        rotation_degrees[
            tick_peaks_corrected[i - 1] : tick_peaks_corrected[i]
        ] = rotation_array
    signed_rotation_degrees = rotation_degrees * rotation_on
    image_rotation_degree_per_line = signed_rotation_degrees[lines_start]
    image_rotation_degree_per_line *= -1

    # line clock is more frequent than the rotation ticks, therefore
    # we need to interpolate the rotation degrees for each line
    # t, c, k = splrep(
    #     lines_start, image_rotation_degree_per_line, s=0
    # )
    # interpolated_image_rotation_degree_per_line = BSpline(t, c, k)(
    #     line_clock
    # )

    return image_rotation_degree_per_line, signed_rotation_degrees
