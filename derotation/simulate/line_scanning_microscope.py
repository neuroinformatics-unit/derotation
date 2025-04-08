"""
This module contains the ``Rotator`` class, which is used to simulate the
acquisition of a rotated image stack as if for each line acquired, the sample
was rotated at a given angle in a given center and plane of rotation.
"""

from typing import Optional, Tuple

import numpy as np
import tqdm
from scipy.ndimage import affine_transform


class Rotator:
    """
    The ``Rotator`` aims to imitate the scanning pattern of a multi-photon
    microscope while the speciment is rotating. Currently, it approximates
    the acquisition of a given line as if it was instantaneous, happening
    while the sample was rotated at a given angle.

    It is also possible to simulate the acquisition of a movie from a
    rotation plane that differs from the scanning plane. To achieve this,
    provide the rotation_plane_angle and if you want the orientation as
    well.

    The purpose of the ``Rotator`` object is to imitate the acquisition of
    rotated samples in order to validate the derotation algorithms and in
    the future, build a forward model of the transformation.

    Parameters
    ----------
    angles : np.ndarray
        An array of angles in degrees, representing the rotation of the
        sample at the time of acquisition. The length of the array should
        be equal to the number of lines per frame multiplied by the number
        of frames.
    image_stack : np.ndarray
        The image stack represents the acquired images, as if there was no
        rotation. The shape of the image stack should be (``num_frames``,
        ``num_lines_per_frame``, ``num_pixels_per_line``). In case you want to
        rotate a single frame, provide an (1, ``num_lines_per_frame``,
        ``num_pixels_per_line``) image stack.
    center : Tuple[int, int], optional
        The center of rotation. If None, the center is going to be the
        center of the image, by default None
    rotation_plane_angle : float, optional
        The z angle of the rotation plane in degrees in relation to the
        scanning plane. If 0, the rotation plane is the same as
        the  scanning plane, by default None.
    rotation_plane_orientation : float, optional
        The angle of the rotation plane in the x-y plane in degrees,
        transposed into the rotation plane. If 0, the rotation
        plane is the same as the scanning plane, by default None.
    blank_pixel_val : Optional[float], optional
        The value to fill the blank pixels with. If None, it is going to be
        the minimum value of the image stack, by default None.

    Raises
    ------
    AssertionError (1)
        If the number of angles is not equal to the number of lines
        per frame multiplied by the number of frames
    AssertionError (2)
        If rotation_plane_orientation is provided, but ``rotation_plane_angle``
        is not provided.
    """

    def __init__(
        self,
        angles: np.ndarray,
        image_stack: np.ndarray,
        center_offset: Tuple[int, int] = (0, 0),
        rotation_plane_angle: float = 0,
        rotation_plane_orientation: float = 0,
        blank_pixel_val: Optional[float] = None,
    ) -> None:
        """Initializes the ``Rotator`` object."""
        #  there should be one angle per line pe frame
        assert len(angles) == image_stack.shape[0] * image_stack.shape[1], (
            f"Number of angles ({len(angles)}) should be equal to the number "
            + "of lines per frame multiplied by the number of frames "
            + f"({image_stack.shape[0] * image_stack.shape[1]})"
        )

        if rotation_plane_orientation is not None:
            # The rotation plane angle makes sense only if the orientation is
            # provided, as without it the rotation is going to be circular
            # and not elliptical.
            assert rotation_plane_angle is not None, (
                "If rotation_plane_orientation is provided, "
                + "rotation_plane_angle should be provided as well."
            )

        self.image_stack = image_stack
        self.num_lines_per_frame = image_stack.shape[1]

        self.pre_homography_center = np.array(
            [image_stack.shape[1] // 2, image_stack.shape[2] // 2]
        )
        if center_offset != (0, 0):
            self.pre_homography_center += np.array(center_offset)

        self.rotation_plane_angle = np.deg2rad(rotation_plane_angle)
        self.rotation_plane_orientation = rotation_plane_orientation
        if rotation_plane_angle == 0:
            self.ps: int = 0
            self.image_size = image_stack.shape[1]
        else:
            self.create_homography_matrices()
            self.image_size = image_stack.shape[1]
            self.ps = 0

        self.post_homography_center = np.array(
            [self.image_size // 2, self.image_size // 2]
        )
        if center_offset != (0, 0):
            self.post_homography_center += np.array(center_offset)

        #  reshape the angles to the shape of the image
        #  stack to ease indexing
        self.angles = angles[: image_stack.shape[0] * self.image_size].reshape(
            image_stack.shape[0], self.image_size
        )

        if blank_pixel_val is None:
            self.blank_pixel_val = self.get_blank_pixels_value()
        else:
            self.blank_pixel_val = blank_pixel_val

    def create_homography_matrices(self) -> None:
        """
        Create the homography matrices to simulate the acquisition of a
        rotated sample from a different plane than the scanning plane.

        The homography matrix is used to transform the image from the scanning
        plane to the rotation plane and vice-versa.

        The homography matrix is defined as:
        H = [[1, 0, 0], [0, cos(theta), 0], [0, 0, 1]]
        where theta is the rotation plane angle in degrees.

        Currently, we are using only the inverse homography matrix to transform
        the image from the rotation plane to the scanning plane.
        """
        #  from the scanning plane to the rotation plane
        self.homography_matrix = np.array(
            [
                [1, 0],
                [0, np.cos(self.rotation_plane_angle)],
            ]
        )

        #  from the rotation plane to the scanning plane
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)

    def calculate_pixel_shift(self) -> None:
        """
        Calculate the pixel shift and the new image size based on the rotation
        plane angle.
        """
        #  store pixels shift based on inverse homography
        line_length = self.image_stack.shape[2]
        self.ps = line_length - np.round(
            np.abs(line_length * np.cos(self.rotation_plane_angle))
        ).astype(int)

        #  round to the nearest even number
        self.ps += self.ps % 2

        #  final image size depends on the pixel shift
        self.image_size = line_length - self.ps

    def rotate_by_line(self) -> np.ndarray:
        """Simulate the acquisition of a rotated image stack as if for each
        line acquired, the sample was rotated at a given angle in a given
        center and plane of rotation.

        Each frame is rotated ``n_lines_per_frame`` times, where
        ``n_lines_per_frame`` is the number of lines per frame in the image
        stack.

        Returns
        -------
        np.ndarray
            The rotated image stack of the same shape as the input image stack,
            i.e. (``num_frames``, ``num_lines_per_frame``,
            ``num_pixels_per_line``).
        """
        rotated_image_stack = np.empty(
            (self.image_stack.shape[0], self.image_size, self.image_size),
            dtype=np.float64,
        )

        for i, image in tqdm.tqdm(
            enumerate(self.image_stack), total=self.image_stack.shape[0]
        ):
            if np.all(
                # don't bother if rotation is less than 0.01 degrees
                np.isclose(self.angles[i], 0, atol=1e-2)
            ):
                rotated_image_stack[i] = image
                continue

            for j, angle in enumerate(self.angles[i]):
                if angle == 0:
                    rotated_image_stack[i][j] = image[j]
                else:
                    # rotate the whole image by the angle
                    rotated_image = self.rotate_sample(
                        image, angle, center=self.post_homography_center
                    )

                    # if the rotation plane angle is not 0,
                    # apply the homography
                    if self.rotation_plane_angle != 0:
                        rotated_image = (
                            self.homography_rotation_to_scanning_plane(
                                rotated_image
                            )
                        )
                        rotated_image = rotated_image

                    # store the rotated image line
                    rotated_image_stack[i][j] = rotated_image[j]

        return rotated_image_stack

    def homography_rotation_to_scanning_plane(
        self, image: np.ndarray
    ) -> np.ndarray:
        """Apply the homography to the image to simulate the acquisition of a
        rotated sample from a different plane than the scanning plane.

        Parameters
        ----------
        image : np.ndarray
            The image to apply the homography.

        Returns
        -------
        np.ndarray
            The transformed image.
        """

        #  rotate the image according to the rotation plane orientation
        if self.rotation_plane_orientation != 0:
            image = self.rotate_sample(
                image,
                self.rotation_plane_orientation,
                center=self.pre_homography_center,
            )

        offset = (
            self.pre_homography_center
            - self.inverse_homography_matrix @ self.pre_homography_center
        )

        # forward transformation
        image = affine_transform(
            image,
            self.inverse_homography_matrix,
            offset=offset,
            output_shape=image.shape,
            order=0,
            mode="constant",
            cval=self.blank_pixel_val,
        )

        #  rotate the image back to the original orientation
        if self.rotation_plane_orientation != 0:
            image = self.rotate_sample(
                image,
                -self.rotation_plane_orientation,
                center=self.pre_homography_center,
            )

        return image

    def rotate_sample(
        self,
        image: np.ndarray,
        angle: float,
        center: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Rotate the entire image by a given angle. Uses affine transformation
        with no interpolation.

        Parameters
        ----------
        image : np.ndarray
            The image to rotate.
        angle : float
            The angle in degrees to rotate the image in degrees. If positive,
            rotates clockwise. If negative, rotates counterclockwise.
        blank_pixel : float
            The value to fill the blank pixels with.

        Returns
        -------
        np.ndarray
            The rotated image.
        """
        # Compute rotation in radians
        angle_rad = np.deg2rad(angle)
        cos, sin = np.cos(angle_rad), np.sin(angle_rad)

        # Rotation matrix clockwise if angle is positive
        rotation_matrix = np.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

        # Compute offset so rotation is around the center
        offset = center - rotation_matrix @ center

        # Apply affine transformation
        rotated_image = affine_transform(
            image,
            rotation_matrix,
            offset=offset,
            output_shape=image.shape,  # Keep original shape
            order=0,
            mode="constant",  # NO interpolation
            cval=self.blank_pixel_val,
        )

        return rotated_image

    def get_blank_pixels_value(self) -> float:
        """Get a default value to fill the edges of the rotated image.
        This is necessary because we are using affine transformation with no
        interpolation, so we need to fill the blank pixels with some value.

        As for now, it returns the minimum value of the image stack.

        Returns
        -------
        float
            The minimum value of the image stack.
        """
        return np.min(self.image_stack)
