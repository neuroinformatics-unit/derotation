import copy
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform


class Rotator:
    def __init__(
        self,
        angles: np.ndarray,
        image_stack: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        rotation_plane_angle: Optional[float] = None,
    ) -> None:
        """Initializes the Rotator object.
        The Rotator aims to imitate a the scanning pattern of a multi-photon
        microscope while the speciment is rotating. Currently, it approximates
        the acquisition of a given line as if it was instantaneous, happening
        while the sample was rotated at a given angle.

        The purpouse of the Rotator object is to imitate the acquisition of
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
            rotation. The shape of the image stack should be (num_frames,
            num_lines_per_frame, num_pixels_per_line). In case you want to
            rotate a single frame, provide an (1, num_lines_per_frame,
            num_pixels_per_line) image stack.

        Raises
        ------
        AssertionError
            If the number of angles is not equal to the number of lines
            per frame multiplied by the number of frames
        """
        #  there should be one angle per line pe frame
        assert len(angles) == image_stack.shape[0] * image_stack.shape[1], (
            f"Number of angles ({len(angles)}) should be equal to the number "
            + "of lines per frame multiplied by the number of frames "
            + f"({image_stack.shape[0] * image_stack.shape[1]})"
        )

        #  reshape the angles to the shape of the image stack to ease indexing
        self.angles = angles.reshape(
            image_stack.shape[0], image_stack.shape[1]
        )
        self.image_stack = image_stack
        self.num_lines_per_frame = image_stack.shape[1]

        if center is None:
            self.center = np.array(image_stack.shape[1:]) / 2
        else:
            self.center = np.array(center)
        
        if rotation_plane_angle is None:
            self.rotation_plane_angle = 0
        else:
            self.rotation_plane_angle = rotation_plane_angle
            self.create_homography_matrices()
    
    def create_homography_matrices(self) -> None:
        #  expansion
        self.homography_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.radians(self.rotation_plane_angle)), 0],
                [0, 0, 1],
            ]
        )

        #  contraction
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)
        

    def rotate_by_line(self) -> np.ndarray:
        """Simulate the acquisition of a rotated image stack as if for each
        line acquired, the sample was rotated at a given angle.

        Each frame is rotated n_lines_per_frame times, where n_lines_per_frame
        is the number of lines per frame in the image stack.

        Returns
        -------
        np.ndarray
            The rotated image stack of the same shape as the input image stack,
            i.e. (num_frames, num_lines_per_frame, num_pixels_per_line).
        """
        rotated_image_stack = copy.deepcopy(self.image_stack)

        for i, image in enumerate(self.image_stack):
            is_this_frame_rotating = not np.all(
                # don't bother if rotation is less than 0.01 degrees
                np.isclose(self.angles[i], 0, atol=1e-2)
            )
            if is_this_frame_rotating:
                for j, angle in enumerate(self.angles[i]):
                    if angle == 0:
                        continue
                    else:
                        rotated_frame = self.rotate(image, angle)
                        rotated_image_stack[i][j] = rotated_frame[j]

        return rotated_image_stack
    
    def apply_homography(self, image: np.ndarray, direction: str) -> np.ndarray:
        if direction == "expand":
            return affine_transform(
                image,
                self.homography_matrix,
                offset=self.center - self.center,
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=self.get_blank_pixels_value(),
            )
        elif direction == "contract":
            return affine_transform(
                image,
                self.inverse_homography_matrix,
                offset=self.center - self.center,
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=self.get_blank_pixels_value(),
            )


    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
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
        if self.rotation_plane_angle != 0:
            image = self.apply_homography(image, "expand")


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
        offset = self.center - rotation_matrix @ self.center

        # Apply affine transformation
        rotated_image = affine_transform(
            image,
            rotation_matrix,
            offset=offset,
            output_shape=image.shape,  # Keep original shape
            order=0,
            mode="constant",  # NO interpolation
            cval=self.get_blank_pixels_value(),
        )

        if self.rotation_plane_angle != 0:
            rotated_image = self.apply_homography(rotated_image, "contract")

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
        return 0 # np.min(self.image_stack)
