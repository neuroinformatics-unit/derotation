from pathlib import Path

import numpy as np
from tifffile import imsave

from derotation.analysis.derotation_pipeline import DerotationPipeline
from derotation.analysis.rotate_images import rotate_frames_line_by_line
from derotation.analysis.saving import add_circle_mask

# ==============================================================================
# PREPROCESSING PIPELINE FOR DEROTATION
# ==============================================================================
dataset = "CAA_1"
pipeline = DerotationPipeline(dataset_name=dataset)
print(f"Dataset {dataset} loaded")
pipeline.process_analog_signals()

# ==============================================================================
# ROTATE THE IMAGE TO THE CORRECT POSITION
# ==============================================================================

rotated_images = rotate_frames_line_by_line(
    pipeline.images_stack, pipeline.image_rotation_degrees_line
)

# ==============================================================================
# MASK THE IMAGE AND SAVE IT
# ==============================================================================
masked = add_circle_mask(rotated_images)

path = Path(f"derotation/data/processed/{dataset}")
path.mkdir(parents=True, exist_ok=True)
imsave(
    f"derotation/data/processed/{dataset}/masked.tif",
    np.array(masked),
)
print(f"Masked image saved in {path}")
