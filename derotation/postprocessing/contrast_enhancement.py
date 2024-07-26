from pathlib import Path

import numpy as np
import tifffile as tf
from skimage.exposure import rescale_intensity


def whole_video_contrast_enhancment(
    tiff_path: Path, saturated_percentage: float
):
    tiff = tf.imread(tiff_path)
    v_min, v_max = np.percentile(
        tiff, (saturated_percentage, 100 - saturated_percentage)
    )

    CE_tiff_path = np.zeros_like(tiff)
    for i, frame in enumerate(tiff):
        CE_tiff_path[i] = rescale_intensity(frame, in_range=(v_min, v_max))

    return CE_tiff_path


def frame_by_frame_contrast_enhancement(
    tiff_path: Path, saturated_percentage: float
):
    tiff = tf.imread(tiff_path)

    CE_tiff_path = np.zeros_like(tiff)
    for i, frame in enumerate(tiff):
        v_min, v_max = np.percentile(
            frame, (saturated_percentage, 100 - saturated_percentage)
        )
        CE_tiff_path[i] = rescale_intensity(frame, in_range=(v_min, v_max))

    return CE_tiff_path
