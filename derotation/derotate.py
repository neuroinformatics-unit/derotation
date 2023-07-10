from pathlib import Path

from adjust_rotation_degrees import (
    apply_new_rotations,
    get_optimal_rotation_degs,
)
from analog_preprocessing import (
    apply_rotation_direction,
    check_number_of_rotations,
    find_rotation_for_each_frame_from_motor,
    get_missing_frames,
    get_starting_and_ending_frames,
    when_is_rotation_on,
)
from get_data import get_data
from scipy.signal import find_peaks

from derotation.plots import (
    analog_signals_overview_plots,
    derotation_video_with_rotation_plot,
    plot_drift_of_centroids,
    threshold_boxplot,
)
from derotation.rotate_images import rotate_images

# ==============================================================================
# PREPROCESSING PIPELINE FOR DEROTATION
# ==============================================================================
(
    image,
    frame_clock,
    line_clock,
    full_rotation,
    rotation_ticks,
    dt,
    config,
    direction,
) = get_data(Path("/Users/laura/data/230327_pollen"))
rot_deg = 360
missing_frames, diffs = get_missing_frames(frame_clock)
frames_start, frames_end, threshold = get_starting_and_ending_frames(
    frame_clock, image
)
#  find the peaks of the rot_tick2 signal
rotation_ticks_peaks = find_peaks(
    rotation_ticks,
    height=4,
    distance=20,
)[0]

check_number_of_rotations(rotation_ticks_peaks, direction, rot_deg, dt)
rotation_on = when_is_rotation_on(full_rotation)
rotation_on = apply_rotation_direction(rotation_on, direction)
(
    image_rotation_degree_per_frame,
    signed_rotation_degrees,
) = find_rotation_for_each_frame_from_motor(
    frame_clock, rotation_ticks_peaks, rotation_on, frames_start
)


# ==============================================================================
# TRY TO OPTIMIZE THE ROTATION DEGREES
# ==============================================================================
opt_result, indexes, optimized_parameters = get_optimal_rotation_degs(
    image, image_rotation_degree_per_frame
)
new_image_rotation_degree_per_frame = apply_new_rotations(
    opt_result, image_rotation_degree_per_frame, indexes
)


# ==============================================================================
# ROTATE THE IMAGE TO THE CORRECT POSITION
# ==============================================================================
(
    rotated_image,
    rotated_image_corrected,
    centers,
    centers_rotated,
    centers_rotated_corrected,
) = rotate_images(
    image,
    image_rotation_degree_per_frame,
    new_image_rotation_degree_per_frame,
)

# ==============================================================================
# PLOTS
# ==============================================================================

centroid_fig = plot_drift_of_centroids(
    centers, centers_rotated, centers_rotated_corrected
)
rotation_video_with_plot = derotation_video_with_rotation_plot(
    rotated_image,
    image,
    rotated_image_corrected,
    centers,
    centers_rotated,
    centers_rotated_corrected,
    frames_start,
    signed_rotation_degrees,
    image_rotation_degree_per_frame,
)

boxplot = threshold_boxplot(diffs, threshold)

analog_overview = analog_signals_overview_plots(
    diffs,
    frame_clock,
    frames_start,
    frames_end,
    line_clock,
    full_rotation,
    rotation_on,
    rotation_ticks,
    rotation_ticks_peaks,
)
