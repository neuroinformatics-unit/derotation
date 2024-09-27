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
    ) -> None:
        """Initializes the Rotator object.

        Parameters
        ----------
        angles : np.ndarray
            The rotation angles by line per frame
        image_stack : np.ndarray
            The image stack to be rotated

        Raises
        ------
        AssertionError
            If the number of angles is not equal to the number of lines
            per frame multiplied by the number of frames
        """
        #  there should be one angle per line pe frame
        assert len(angles) == image_stack.shape[0] * image_stack.shape[1]

        self.angles = angles
        self.image_stack = image_stack
        self.num_lines_per_frame = image_stack.shape[1]

        if center is None:
            self.center = np.array(image_stack.shape[1:]) / 2
        else:
            self.center = np.array(center)

    def rotate_by_line(self) -> np.ndarray:
        """Rotates the image stack line by line, using the rotation angles
        provided.

        Returns
        -------
        np.ndarray
            The rotated image stack.
        """
        rotated_image_stack = copy.deepcopy(self.image_stack)
        self.get_blank_pixels_value()

        for i, image in enumerate(self.image_stack):
            start_angle_idx = self.angles[i * self.num_lines_per_frame]
            end_angle_idx = self.angles[self.num_lines_per_frame * (i + 1) - 1]

            is_this_frame_rotating = np.any(
                np.abs(self.angles[start_angle_idx:end_angle_idx]) > 0.00001
            )
            if is_this_frame_rotating:
                frame = copy.deepcopy(image)
                for j, angle in enumerate(
                    range(start_angle_idx, end_angle_idx)
                ):
                    if angle == 0:
                        continue
                    else:
                        rotated_frame = self.rotate(image, angle)
                        frame[j] = rotated_frame[j]
                rotated_image_stack[i] = frame

        return rotated_image_stack

    def rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        # Compute rotation in radians
        angle_rad = np.deg2rad(angle)
        cos, sin = np.cos(angle_rad), np.sin(angle_rad)

        # Rotation matrix
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
            order=0,  # NO interpolation
            mode="constant",
            cval=0,  # Fill empty values with 0 (black)
        )

        return rotated_image

    def get_blank_pixels_value(self) -> float:
        """Returns the minimum value of the image stack.

        Returns
        -------
        float
            The minimum value of the image stack.
        """
        return np.min(self.image_stack)
