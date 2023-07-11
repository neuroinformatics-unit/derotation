import numpy as np
from scipy.ndimage import rotate

from derotation.analysis.find_centroid import find_centroid_pipeline


def rotate_images(
    image,
    image_rotation_degree_per_frame,
    new_image_rotation_degree_per_frame,
):
    #  rotate the image to the correct position according to the frame_degrees
    rotated_image = np.empty_like(image)
    rotated_image_corrected = np.empty_like(image)
    centers = []
    centers_rotated = []
    centers_rotated_corrected = []
    for i in range(len(image)):
        lower_threshold = -2700
        higher_threshold = -2600
        binary_threshold = 32
        sigma = 2.5

        defoulting_parameters = [
            lower_threshold,
            higher_threshold,
            binary_threshold,
            sigma,
        ]

        rotated_image[i] = rotate(
            image[i], image_rotation_degree_per_frame[i], reshape=False
        )
        rotated_image_corrected[i] = rotate(
            image[i], new_image_rotation_degree_per_frame[i], reshape=False
        )

        # params = optimized_parameters[i]
        # if i in indexes else defoulting_parameters

        centers.append(find_centroid_pipeline(image[i], defoulting_parameters))
        centers_rotated.append(
            find_centroid_pipeline(rotated_image[i], defoulting_parameters)
        )
        centers_rotated_corrected.append(
            find_centroid_pipeline(
                rotated_image_corrected[i], defoulting_parameters
            )
        )

    return (
        rotated_image,
        rotated_image_corrected,
        centers,
        centers_rotated,
        centers_rotated_corrected,
    )
