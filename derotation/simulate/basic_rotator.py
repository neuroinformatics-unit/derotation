import copy

import numpy as np
from scipy.ndimage import affine_transform


class Rotator:
    def __init__(self, angles: np.ndarray, image_stack: np.ndarray) -> None:
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
        blank_pixel = self.get_blank_pixels_value()

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
                        rotated_frame = self.rotate(image, angle, blank_pixel)
                        rotated_image_stack[i][j] = rotated_frame[j]

        return rotated_image_stack

    @staticmethod
    def rotate(
        image: np.ndarray, angle: float, blank_pixel: float
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

        # Calculate center of the image
        center_y, center_x = np.array(image.shape) / 2

        # Rotation matrix clockwise if angle is positive
        rotation_matrix = np.array(
            [
                [cos, -sin],
                [sin, cos],
            ]
        )

        # Compute offset so rotation is around the center
        offset = np.array([center_y, center_x]) - rotation_matrix @ np.array(
            [center_y, center_x]
        )

        # Apply affine transformation
        rotated_image = affine_transform(
            image,
            rotation_matrix,
            offset=offset,
            output_shape=image.shape,  # Keep original shape
            order=0,
            mode="constant",  # NO interpolation
            cval=blank_pixel,
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
