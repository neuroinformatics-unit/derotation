import copy
from typing import Optional, Tuple

import numpy as np
import tqdm
from scipy.ndimage import affine_transform


def derotate_an_image_array_line_by_line(
    image_stack: np.ndarray,
    rot_deg_line: np.ndarray,
    blank_pixels_value: float = 0,
    center: Optional[Tuple[int, int]] = None,
    plotting_hook_line_addition=None,
    plotting_hook_image_completed=None,
    use_homography: bool = False,
    rotation_plane_angle: Optional[int] = None,
    rotation_plane_orientation: Optional[int] = None,
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
        The center of rotation (x, y). If not provided, defaults to the
        center of the image.

    Returns
    -------
    np.ndarray
        The derotated image stack.
    """

    num_lines_per_frame = image_stack.shape[1]

    if center is None:
        #  assumes square images
        center = (
            image_stack.shape[2] // 2,
            image_stack.shape[1] // 2,
        )  # Default center of rotation
    #  Swap x and y and reshape to column vector
    center = np.array(center[::-1]).reshape(2, 1)

    if use_homography:
        #  derotation should happen in the plane in which the rotation is circular.
        #  scanning to rotation plane
        homography_matrix = np.array(
            [
                [1, 0],
                # shear happens in the x direction (columns)
                [0, np.cos(np.deg2rad(rotation_plane_angle))],
            ]
        )
        # rotation matrix for plane orientation
        # invert the angle to get the correct rotation
        angle_rad = np.deg2rad(rotation_plane_orientation)
        cos, sin = np.cos(angle_rad), np.sin(angle_rad)

        orientation = np.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

        h_offset = center.T[0] - homography_matrix @ center.T[0]
        o_offset = center.T[0] - orientation @ center.T[0]
        io_offset = center.T[0] - orientation.T @ center.T[0]

        #  final image shape
        # shape = (image_stack.shape[1], int(image_stack.shape[2] / np.cos(np.radians(rotation_plane_angle))))

        #  shear the image stack (i.e. shift it to the rotation plane)
        hom_image_stack = []
        for i, image in enumerate(image_stack):
            # fig, ax = plt.subplots()
            # ax.imshow(image, cmap='gray')
            # plt.savefig(f'./debug/original_{i}.png')

            # first rotate according to orientation
            transformed = affine_transform(
                image,
                orientation,
                offset=o_offset,
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=blank_pixels_value,
            )

            # fig, ax = plt.subplots()
            # ax.imshow(transformed, cmap='gray')
            # plt.savefig(f'./debug/orientation_{i}.png')

            # then apply homography
            transformed = affine_transform(
                transformed,
                homography_matrix,
                offset=h_offset,
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=blank_pixels_value,
            )

            # fig, ax = plt.subplots()
            # ax.imshow(transformed, cmap='gray')
            # plt.savefig(f'./debug/homography_{i}.png')

            # rotate it back to the original orientation
            transformed = affine_transform(
                transformed,
                orientation.T,
                offset=io_offset,
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=blank_pixels_value,
            )

            # fig, ax = plt.subplots()
            # ax.imshow(transformed, cmap='gray')
            # plt.savefig(f'./debug/inv_orientation_{i}.png')

            #  clip the image to the original size
            # offset = int((transformed.shape[1] - image.shape[1]) / 2)
            # transformed = transformed[:, offset : offset + image.shape[1]]
            hom_image_stack.append(transformed)

        hom_image_stack = np.asarray(hom_image_stack)
        #  check shape
        assert (
            hom_image_stack.shape == image_stack.shape
        ), f"Shape mismatch: {hom_image_stack.shape} != {image_stack.shape}"
        image_stack = hom_image_stack

    derotated_image_stack = copy.deepcopy(image_stack)

    previous_image_completed = True
    rotation_completed = True

    rot_deg_line = rot_deg_line[: num_lines_per_frame * len(image_stack)]

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

            # Center the coordinates
            centered_coords = line_coords - center

            # Apply rotation matrix
            rotated_coords = rotation_matrix @ centered_coords

            # Shift back the rotated coordinates to the image space
            final_coords = (rotated_coords + center).astype(int)

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
            
            empty_image = np.ones_like(derotated_filled_image) * np.nan
            empty_image[
                final_coords[0][valid_mask], final_coords[1][valid_mask]
            ] = line[valid_mask]

            if plotting_hook_line_addition is not None:
                plotting_hook_line_addition(
                    derotated_filled_image,
                    empty_image,
                    image_counter,
                    line_counter,
                    angle,
                    image_stack[image_counter],
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

    # if use_homography:
    #     #  rotation plane to scanning plane transformation
    #     inverse_homography_matrix = np.linalg.inv(homography_matrix)
    #     inverse_orientation = np.linalg.inv(orientation)

    #     for i, img in enumerate(derotated_image_stack):
    #         # first apply homography
    #         transformed = affine_transform(
    #             img,
    #             inverse_homography_matrix,
    #             # offset=center,
    #             output_shape=image.shape,
    #             order=0,
    #             mode="constant",
    #             cval=blank_pixels_value,
    #         )
    #         # then apply orientation
    #         # transformed = affine_transform(
    #         #     transformed,
    #         #     inverse_orientation,
    #         #     # offset=center,
    #         #     output_shape=image.shape,
    #         #     order=0,
    #         #     mode="constant",
    #         #     cval=blank_pixels_value,
    #         # )
    #         derotated_image_stack[i] = transformed

    return derotated_image_stack
