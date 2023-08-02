from derotation.analysis.derotation_pipeline import DerotationPipeline
from derotation.analysis.rotate_images import rotate_frames_line_by_line

# ==============================================================================
# PREPROCESSING PIPELINE FOR DEROTATION
# ==============================================================================
pipeline = DerotationPipeline()
pipeline.process_analog_signals()

# ==============================================================================
# ROTATE THE IMAGE TO THE CORRECT POSITION
# ==============================================================================

rotate_frames_line_by_line(
    pipeline.images_stack, pipeline.image_rotation_degrees_line
)
