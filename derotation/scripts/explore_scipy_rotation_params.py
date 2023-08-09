import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.ndimage import rotate

from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline()
pipeline.process_analog_signals()


scipy_rotate_keys = [
    "constant",
    "grid-constant",
    "nearest",
    "reflect",
    "mirror",
    "wrap",
    "grid-wrap",
]
order = [0, 1, 3, 4, 5]


def rotate_with_given_flavour(
    image_stack, rotation_degrees, key, order, height
):
    previous_image_completed = True
    rotation_completed = True

    this_flavour = []
    for i, rotation in enumerate(rotation_degrees):
        line_counter = i % height
        image_counter = i // height
        is_rotating = np.absolute(rotation) > 0.00001
        image_scanning_completed = line_counter == (height - 1)
        rotation_just_finished = not is_rotating and (
            np.absolute(rotation_degrees[i - 1]) > np.absolute(rotation)
        )

        if is_rotating:
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
                order=order,
                mode=key,
            )
            rotated_mask = rotate(
                empty_image_mask,
                rotation,
                reshape=False,
                order=order,
                mode=key,
            )

            #  apply rotated mask to rotated line-image
            masked = ma.masked_array(rotated_line, rotated_mask)

            if previous_image_completed:
                rotated_filled_image = np.zeros_like(img_with_new_lines)

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
            # image_stack[image_counter] = rotated_filled_image
            this_flavour.append(rotated_filled_image)
            previous_image_completed = True

            print("Image {} rotated".format(image_counter))

            if rotation_just_finished:
                # rotation_completed = True
                break

    return this_flavour


def rotate_frames_line_by_line_explorative(image_stack, rotation_degrees):
    #  fill new_rotated_image_stack with non-rotated images first
    _, height, _ = image_stack.shape

    rotated_flavours = {}
    for key in scipy_rotate_keys:
        for o in order:
            rotated_flavours[(key, o)] = rotate_with_given_flavour(
                image_stack, rotation_degrees, key, o, height
            )
        print("Flavour {} completed".format(key))

    return rotated_flavours


rotated_flavours = rotate_frames_line_by_line_explorative(
    pipeline.images_stack, pipeline.image_rotation_degrees_line
)


for j, key in enumerate(scipy_rotate_keys):
    for k, o in enumerate(order):
        len_images = len(rotated_flavours)
        row = 3
        col = 7

        fig, ax = plt.subplots(row, col)
        for i in range(len_images):
            try:
                tup = (i // col, i % col)
                ax[tup].imshow(rotated_flavours[(key, o)][i])
                ax[tup].axis("off")
            except IndexError:
                print(i, key, o)
        plt.subplots_adjust(wspace=0, hspace=0)
        #  set title
        plt.suptitle("key: {}, order: {}".format(key, o))
        plt.savefig(
            "derotation/rotate_flavours/{}_{}.png".format(key, o), dpi=500
        )


with open("derotation/rotate_flavours/rotated_flavours.pkl", "wb") as f:
    pickle.dump(rotated_flavours, f)
