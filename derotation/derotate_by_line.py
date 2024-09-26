import copy

import numpy as np
import tqdm
from scipy.ndimage import rotate


def derotate_an_image_array_line_by_line(
    image_stack: np.ndarray,
    rot_deg_line: np.ndarray,
    blank_pixels_value: float = 0,
    num_lines_per_frame=None,
    plotting_hook_line_addition=None,
    plotting_hook_image_completed=None,
) -> np.ndarray:
    """Rotates the image stack line by line, using the rotation angles
    provided.

    Description of the algorithm:
    - takes one line from the image stack
    - creates a new image with only that line
    - rotates the line by the given angle
    - substitutes the line in the new image
    - adds the new image to the rotated image stack

    Edge cases and how they are handled:
    - the rotation starts in the middle of the image -> the previous lines
    are copied from the first frame
    - the rotation ends in the middle of the image -> the remaining lines
    are copied from the last frame

    Parameters
    ----------
    image_stack : np.ndarray
        The image stack to be rotated.
    rot_deg_line : np.ndarray
        The rotation angles by line.

    Returns
    -------
    np.ndarray
        The rotated image stack.
    """
    if num_lines_per_frame is None:
        num_lines_per_frame = image_stack.shape[1]
    rotated_image_stack = copy.deepcopy(image_stack)
    previous_image_completed = True
    rotation_completed = True

    for i, rotation in tqdm.tqdm(
        enumerate(rot_deg_line), total=len(rot_deg_line)
    ):
        line_counter = i % num_lines_per_frame
        image_counter = i // num_lines_per_frame

        is_rotating = np.absolute(rotation) > 0.00001
        image_scanning_completed = line_counter == (num_lines_per_frame - 1)
        if i == 0:
            rotation_just_finished = False
        else:
            rotation_just_finished = not is_rotating and (
                np.absolute(rot_deg_line[i - 1]) > np.absolute(rotation)
            )

        if is_rotating:
            if rotation_completed and (line_counter != 0):
                # when starting a new rotation in the middle of the image
                rotated_filled_image = (
                    np.ones_like(image_stack[image_counter])
                    * blank_pixels_value
                )  # non sampled pixels are set to the min val of the image
                rotated_filled_image[:line_counter] = image_stack[
                    image_counter
                ][:line_counter]
            elif previous_image_completed:
                rotated_filled_image = (
                    np.ones_like(image_stack[image_counter])
                    * blank_pixels_value
                )

            rotation_completed = False

            img_with_new_lines = image_stack[image_counter]
            line = img_with_new_lines[line_counter]

            image_with_only_line = np.zeros_like(img_with_new_lines)
            image_with_only_line[line_counter] = line

            rotated_line = rotate(
                image_with_only_line,
                rotation,
                reshape=False,
                order=0,
                mode="constant",
            )

            rotated_filled_image = np.where(
                rotated_line == 0, rotated_filled_image, rotated_line
            )
            previous_image_completed = False

            if plotting_hook_line_addition is not None:
                plotting_hook_line_addition(
                    rotated_filled_image,
                    rotated_line,
                    image_counter,
                    line_counter,
                    rotation,
                )
        if (
            image_scanning_completed and not rotation_completed
        ) or rotation_just_finished:
            if rotation_just_finished:
                rotation_completed = True

                rotated_filled_image[line_counter + 1 :] = image_stack[
                    image_counter
                ][line_counter + 1 :]

            rotated_image_stack[image_counter] = rotated_filled_image
            previous_image_completed = True

            if plotting_hook_image_completed is not None:
                plotting_hook_image_completed(
                    rotated_image_stack, image_counter
                )

    return rotated_image_stack
