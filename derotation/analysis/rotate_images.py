import numpy as np
import numpy.ma as ma
from scipy.ndimage import rotate


def rotate_frames_line_by_line(image_stack, rotation_degrees):
    #  fill new_rotated_image_stack with non-rotated images first
    num_images, height, width = image_stack.shape

    previous_image_completed = True
    rotation_completed = True
    for i, rotation in enumerate(rotation_degrees):
        line_counter = i % height
        image_counter = i // height
        is_rotating = np.absolute(rotation) > 0.00001
        image_scanning_completed = line_counter == (height - 1)
        rotation_just_finished = not is_rotating and (
            np.absolute(rotation_degrees[i - 1]) > np.absolute(rotation)
        )

        if is_rotating:
            if rotation_completed and (line_counter != 0):
                #  starting a new rotation in the middle of the image
                rotated_filled_image = image_stack[image_counter]
            elif previous_image_completed:
                # rotation in progress and new image to be rotated
                # rotated_filled_image = np.zeros_like(
                #     image_stack[image_counter]
                # )
                rotated_filled_image = image_stack[image_counter]

            rotation_completed = False

            #  we want to take the line from the row image collected
            img_with_new_lines = image_stack[image_counter]
            line = img_with_new_lines[line_counter]

            image_with_only_line = np.zeros_like(img_with_new_lines)
            image_with_only_line[line_counter] = line

            empty_image_mask = np.ones_like(img_with_new_lines)
            empty_image_mask[line_counter] = 0

            rotated_line = rotate(
                image_with_only_line,
                rotation,
                reshape=False,
                order=3,
                mode="constant",
            )
            rotated_mask = rotate(
                empty_image_mask,
                rotation,
                reshape=False,
                order=3,
                mode="constant",
            )

            #  apply rotated mask to rotated line-image
            masked = ma.masked_array(rotated_line, rotated_mask)

            #  substitute the non masked values in the new image
            rotated_filled_image = np.where(
                masked.mask, rotated_filled_image, masked.data
            )
            previous_image_completed = False
            print("*", end="")

        if (
            image_scanning_completed
            # and there_is_a_rotated_image_in_locals
            and not rotation_completed
        ) or rotation_just_finished:
            if rotation_just_finished:
                rotation_completed = True
                #  add missing lines at the end of the image
                rotated_filled_image[line_counter + 1 :] = image_stack[
                    image_counter
                ][line_counter + 1 :]

            image_stack[image_counter] = rotated_filled_image
            previous_image_completed = True

            print("Image {} rotated".format(image_counter))

    return image_stack
