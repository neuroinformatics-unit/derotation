import copy
from typing import Optional, Tuple

import numpy as np
import tqdm


def derotate_an_image_array_line_by_line(
    image_stack: np.ndarray,
    rot_deg_line: np.ndarray,
    blank_pixels_value: float = 0,
    center: Optional[Tuple[int, int]] = None,
    plotting_hook_line_addition=None,
    plotting_hook_image_completed=None,
) -> np.ndarray:
    """Rotates the image stack line by line, using the rotation angles
    provided.

    Description of the algorithm:
    - takes one line from the image stack
    - creates a new image with only that line
    - rotates the line by the given angle without interpolation
    - substitutes the line in the new image
    - adds the new image to the derotated image stack

    Edge cases and how they are handled:
    - the rotation starts in the middle of the image -> the previous lines
    are copied from the first frame
    - the rotation ends in the middle of the image -> the remaining lines
    are copied from the last frame

    Parameters
    ----------
    image_stack : np.ndarray
        The image stack to be derotated.
    rot_deg_line : np.ndarray
        The rotation angles by line.
    center : tuple, optional
        The center of rotation (y, x). If not provided, defaults to the
        center of the image.

    Returns
    -------
    np.ndarray
        The derotated image stack.
    """

    num_lines_per_frame = image_stack.shape[1]

    if center is None:
        center = (
            image_stack.shape[2] // 2,
            image_stack.shape[1] // 2,
        )  # Default center
    else:
        # invert order
        center = (center[1], center[0])

    derotated_image_stack = copy.deepcopy(image_stack)
    previous_image_completed = True
    rotation_completed = True

    for i, angle in tqdm.tqdm(
        enumerate(rot_deg_line), total=len(rot_deg_line)
    ):
        line_counter = i % num_lines_per_frame
        image_counter = i // num_lines_per_frame

        is_rotating = np.absolute(angle) > 0.00001
        image_scanning_completed = line_counter == (num_lines_per_frame - 1)
        if i == 0:
            rotation_just_finished = False
        else:
            rotation_just_finished = not is_rotating and (
                np.absolute(rot_deg_line[i - 1]) > np.absolute(angle)
            )

        if is_rotating:
            if rotation_completed and (line_counter != 0):
                # when starting a new rotation in the middle of the image
                derotated_filled_image = (
                    np.ones_like(image_stack[image_counter])
                    * blank_pixels_value
                )  # non sampled pixels are set to the min val of the image
                derotated_filled_image[:line_counter] = image_stack[
                    image_counter
                ][:line_counter]
            elif previous_image_completed:
                derotated_filled_image = (
                    np.ones_like(image_stack[image_counter])
                    * blank_pixels_value
                )

            rotation_completed = False

            img_with_new_lines = image_stack[image_counter]
            line = img_with_new_lines[line_counter]

            image_with_only_line = np.zeros_like(img_with_new_lines)
            image_with_only_line[line_counter] = line

            # Rotate the line as a whole vector without interpolation
            angle_rad = np.deg2rad(angle)
            cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)

            # Calculate rotation matrix
            rotation_matrix = np.array(
                [
                    [cos_angle, -sin_angle],
                    [sin_angle, cos_angle],
                ]
            )

            # Line coordinates
            line_length = img_with_new_lines.shape[1]
            x_coords = np.arange(line_length)
            y_coords = np.full_like(x_coords, line_counter)

            # Stack the coordinates into (y, x) pairs
            line_coords = np.vstack((y_coords, x_coords))

            # Center the coordinates relative to the rotation center
            centered_coords = line_coords - np.array(center).reshape(2, 1)

            # Apply rotation matrix
            rotated_coords = rotation_matrix @ centered_coords

            # Shift back the rotated coordinates to the image space
            final_coords = (
                rotated_coords + np.array(center).reshape(2, 1)
            ).astype(int)

            # Valid coordinates that fall within image bounds
            valid_mask = (
                (final_coords[0] >= 0)
                & (final_coords[0] < image_with_only_line.shape[0])
                & (final_coords[1] >= 0)
                & (final_coords[1] < image_with_only_line.shape[1])
            )

            # Place the rotated line in the output image without interpolation
            derotated_filled_image[
                final_coords[0][valid_mask], final_coords[1][valid_mask]
            ] = line[valid_mask]

            previous_image_completed = False

            if plotting_hook_line_addition is not None:
                plotting_hook_line_addition(
                    derotated_filled_image,
                    image_with_only_line,
                    image_counter,
                    line_counter,
                    angle,
                )

        if (
            image_scanning_completed and not rotation_completed
        ) or rotation_just_finished:
            if rotation_just_finished:
                rotation_completed = True

                derotated_filled_image[line_counter + 1 :] = image_stack[
                    image_counter
                ][line_counter + 1 :]

            derotated_image_stack[image_counter] = derotated_filled_image
            previous_image_completed = True

            if plotting_hook_image_completed is not None:
                plotting_hook_image_completed(
                    derotated_image_stack, image_counter
                )

    return derotated_image_stack
