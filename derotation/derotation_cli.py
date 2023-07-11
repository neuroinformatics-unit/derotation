from derotation_pipeline import DerotationPipeline

from derotation.analysis.adjust_rotation_degrees import (
    apply_new_rotations,
    get_optimal_rotation_degs,
)
from derotation.analysis.rotate_images import rotate_images
from derotation.plots.plots import (
    analog_signals_overview_plots,
    derotation_video_with_rotation_plot,
    plot_drift_of_centroids,
    threshold_boxplot,
)

# ==============================================================================
# PREPROCESSING PIPELINE FOR DEROTATION
# ==============================================================================
pipeline = DerotationPipeline()
pipeline.process_analog_signals()


# ==============================================================================
# TRY TO OPTIMIZE THE ROTATION DEGREES
# ==============================================================================
opt_result, indexes, optimized_parameters = get_optimal_rotation_degs(
    pipeline.image, pipeline.image_rotation_degree_per_frame
)
new_image_rotation_degree_per_frame = apply_new_rotations(
    opt_result, pipeline.image_rotation_degree_per_frame, indexes
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
    pipeline.image,
    pipeline.image_rotation_degree_per_frame,
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
    pipeline.image,
    rotated_image_corrected,
    centers,
    centers_rotated,
    centers_rotated_corrected,
    pipeline.frames_start,
    pipeline.signed_rotation_degrees,
    pipeline.image_rotation_degree_per_frame,
)

boxplot = threshold_boxplot(pipeline.diffs, pipeline.threshold)

analog_overview = analog_signals_overview_plots(
    pipeline.diffs,
    pipeline.frame_clock,
    pipeline.frames_start,
    pipeline.frames_end,
    pipeline.line_clock,
    pipeline.full_rotation,
    pipeline.rotation_on,
    pipeline.rotation_ticks,
    pipeline.rotation_ticks_peaks,
)
