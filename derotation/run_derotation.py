from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline()

pipeline.process_analog_signals()

rotated_images = pipeline.rotate_frames_line_by_line()

masked = pipeline.add_circle_mask(rotated_images)
pipeline.save(masked)
