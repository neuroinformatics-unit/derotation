import numpy as np
import numpy.ma as ma
from scipy.ndimage import rotate

from derotation.analysis.find_centroid import find_centroid_pipeline


def image_stack_rotation(image_stack, rotation_degrees):
    rotated_image_stack = np.empty_like(image_stack)
    for i in range(len(image_stack)):
        rotated_image_stack[i] = rotate(
            image_stack[i], rotation_degrees[i], reshape=False
        )
    return rotated_image_stack


def rotate_frames_line_by_line(image_stack, rotation_degrees):
    #  fill new_rotated_image_stack with non-rotated images first
    num_images, height, width = image_stack.shape

    previous_image_completed = True
    for i, rotation in enumerate(rotation_degrees):
        line_counter = i % height
        image_counter = i // height

        if rotation > 0.00001:
            #  we want to take the line from the row image collected
            img_with_new_lines = image_stack[image_counter]
            line = img_with_new_lines[line_counter]

            image_with_only_line = np.zeros_like(img_with_new_lines)
            image_with_only_line[line_counter] = line

            empty_image_mask = np.ones_like(img_with_new_lines)
            empty_image_mask[line_counter] = 0

            rotated_line = rotate(
                image_with_only_line, rotation, reshape=False
            )
            rotated_mask = rotate(empty_image_mask, rotation, reshape=False)

            #  apply rotated mask to rotated line-image
            masked = ma.masked_array(rotated_line, rotated_mask)

            if previous_image_completed:
                rotated_filled_image = img_with_new_lines

            #  substitute the non masked values in the new image
            rotated_filled_image = np.where(
                masked.mask, rotated_filled_image, masked.data
            )
            previous_image_completed = False
            print("*", end="")

        if (
            line_counter == (height - 1)
            and locals().get("rotated_filled_image", None) is not None
        ) or (rotation < 0.000001 and rotation_degrees[i - 1] > 1):
            #  at the next cycle we have a new image picked
            #  so we can overwrite the previous one with the latest rotated one
            #  or the rotation is starting again we just have an incomplete
            #  rotated image
            image_stack[image_counter] = rotated_filled_image
            previous_image_completed = True

            print("Image {} rotated".format(image_counter))

    return image_stack


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
