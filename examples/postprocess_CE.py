from pathlib import Path

import numpy as np
import tifffile as tf
from skimage.exposure import rescale_intensity

derotated_tiff_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/incremental/derotated/NO_CE/derotated_incremental_image_stack_NO_ce.tif"
)
derotated_tiff = tf.imread(derotated_tiff_path)

saturated_percentage = 0.35
v_min, v_max = np.percentile(
    derotated_tiff, (saturated_percentage, 100 - saturated_percentage)
)

CE_derotated_tiff_path = np.zeros_like(derotated_tiff)
for i, frame in enumerate(derotated_tiff):
    CE_derotated_tiff_path[i] = rescale_intensity(
        frame, in_range=(v_min, v_max)
    )

tf.imwrite(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/incremental/derotated/volume_CE/derotated_incremental_image_stack_rescaled.tif",
    CE_derotated_tiff_path,
)
