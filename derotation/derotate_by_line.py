"""
This module contains the function ``derotate_an_image_array_line_by_line``,
which derotates an image stack line by line using the provided rotation
angles. This is the core function of the derotation pipeline.
In addition to the main function, this module also contains the
``apply_homography`` function, which applies a homography to the image stack
to transform it to the plane of rotation. This is used in the derotation
pipeline to correct for a mismatch between the plane of rotation and the
imaging plane.
"""

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
    rotation_plane_angle: int = 0,
    rotation_plane_orientation: int = 0,
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

    Center of rotation:
        - if not provided, the center of the image is used

    Homography:
        - if use_homography is True, the image stack is first transformed to
          the plane of rotation, then derotated. See apply_homography for more
          details.

    Parameters
    ----------
    image_stack : np.ndarray
        The image stack to be derotated.
    rot_deg_line : np.ndarray
        The rotation angles by line.
    center : tuple, optional
        The center of rotation (x, y). If not provided, defaults to the
        center of the image.
    blank_pixels_value : float, optional
        The value to be used for blank pixels. Defaults to 0.
    plotting_hook_line_addition : callable, optional
        A function that will be called after each line is added to the
        derotated image stack.
    plotting_hook_image_completed : callable, optional
        A function that will be called after each image is completed.
    use_homography : bool, optional
        Whether to use homography to transform the image stack to the plane
        of rotation. Defaults to ``False``.
    rotation_plane_angle : int, optional
        The angle of the plane of rotation. Required if use_homography is
        ``True``.
    rotation_plane_orientation : int, optional
        The orientation of the plane of rotation. Required if
        ``use_homography`` is ``True``.

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
        image_stack = apply_homography(
            rotation_plane_angle,
            rotation_plane_orientation,
            image_stack,
            center,
            blank_pixels_value,
        )

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

            line = image_stack[image_counter][line_counter]

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
            line_length = num_lines_per_frame
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
                & (final_coords[0] < num_lines_per_frame)
                & (final_coords[1] >= 0)
                & (final_coords[1] < num_lines_per_frame)
            )

            # Place the rotated line in the output image without interpolation
            derotated_filled_image[
                final_coords[0][valid_mask], final_coords[1][valid_mask]
            ] = line[valid_mask]

            previous_image_completed = False

            if plotting_hook_line_addition is not None:
                empty_image = np.ones_like(derotated_filled_image) * np.nan
                empty_image[
                    final_coords[0][valid_mask], final_coords[1][valid_mask]
                ] = line[valid_mask]

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

    #  movie remains stretched, as if it was captured in the plane of rotation
    return derotated_image_stack


def apply_homography(
    rotation_plane_angle: int,
    rotation_plane_orientation: int,
    image_stack: np.ndarray,
    center: np.ndarray,
    blank_pixels_value: float,
) -> np.ndarray:
    """Applies a homography to the image stack to transform it to the plane of
    rotation. The homography is applied in three steps:
    1. Rotate the image stack according to the orientation of the plane of
    rotation.
    2. Shear the image stack to the plane of rotation.
    3. Rotate the image stack back to the original orientation.
    Rotation plane angle and orientation are calculated from an external
    source, by fitting an ellipse. The ellipse can have a different
    orientation than the plane of rotation, so these three steps are
    necessary.

    Parameters
    ----------
    rotation_plane_angle : int
        The angle of the plane of rotation in respect to the imaging plane.
    rotation_plane_orientation : int
        The orientation of the plane of rotation, as calculated from the
        ellipse fitting.
    image_stack : np.ndarray
        The image stack to be transformed.
    center : np.ndarray
        The center of rotation.
    blank_pixels_value : float
        The value to be used for blank pixels.

    Returns
    -------
    np.ndarray
        The transformed image stack.
    """
    #  derotation should happen in the plane in which the rotation is circular.
    #  scanning to rotation plane

    # Convert angles to radians
    angle_rad = np.deg2rad(rotation_plane_orientation)
    shear_rad = np.deg2rad(rotation_plane_angle)

    cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
    cos_alpha = np.cos(shear_rad)

    # Rotation matrix for plane orientation
    R = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Shear (homography-like) matrix for scanning into rotation plane
    H = np.array([[1, 0, 0], [0, cos_alpha, 0], [0, 0, 1]])

    # Inverse rotation matrix
    R_inv = np.array(
        [[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    # Compute the combined transformation matrix
    A = R_inv @ H @ R  # Rotation -> Shear -> Inverse rotation

    # Convert `center` into homogeneous coordinates correctly
    center_homogeneous = np.append(center, 1)  # Fix

    # Compute offset
    offset = center_homogeneous - A @ center_homogeneous

    # Adjust transformation matrix to include offset
    A[:2, 2] = offset[:2]

    # Apply single affine transformation
    new_image_stack = np.array(
        [
            affine_transform(
                image,
                A[:2, :2],  # Extract the 2x2 part for transformation
                offset=A[:2, 2],  # Apply the computed offset
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=blank_pixels_value,
            )
            for image in image_stack
        ]
    )

    #  check shape
    assert new_image_stack.shape == image_stack.shape, (
        f"Shape mismatch: {new_image_stack.shape} != {image_stack.shape}"
    )
    image_stack = new_image_stack

    return image_stack
