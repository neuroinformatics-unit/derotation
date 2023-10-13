from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline("full_rotation")
pipeline.assume_full_rotation = False

pipeline.process_analog_signals()

rotated_images = pipeline.rotate_frames_line_by_line()

# rotated_images = pipeline.roatate_by_frame_incremental()

masked = pipeline.add_circle_mask(rotated_images)
pipeline.save(masked)
