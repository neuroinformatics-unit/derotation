import copy

import numpy as np
from scipy.ndimage import rotate


class Rotator:
    def __init__(self, angles: np.ndarray, image_stack: np.ndarray) -> None:
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

    def rotate_by_line(self) -> np.ndarray:
        """Rotates the image stack line by line, using the rotation angles
        provided.

        Returns
        -------
        np.ndarray
            The rotated image stack.
        """
        rotated_image_stack = copy.deepcopy(self.image_stack)
        blank_pixel = self.get_blank_pixels_value()

        for i, image in enumerate(self.image_stack):
            start_angle_idx = self.angles[i]
            end_angle_idx = self.angles[self.num_lines_per_frame * (i + 1) - 1]

            is_this_frame_rotating = np.any(
                np.abs(self.angles[start_angle_idx:end_angle_idx]) > 0.00001
            )
            if is_this_frame_rotating:
                rotated_frame = np.ones_like(image) * blank_pixel

                for j, line in enumerate(image):
                    angle = self.angles[i * self.num_lines_per_frame + j]
                    if np.abs(angle) > 0.00001:
                        image_with_only_line = np.zeros_like(image)
                        image_with_only_line[j] = line

                        rotated_line = rotate(
                            image_with_only_line,
                            angle,
                            reshape=False,
                            order=0,
                            mode="constant",
                        )

                        rotated_frame = np.where(
                            rotated_line == 0, rotated_frame, rotated_line
                        )
                    else:
                        rotated_frame[j] = line

                rotated_image_stack[i] = rotated_frame

        return rotated_image_stack

    def get_blank_pixels_value(self) -> float:
        """Returns the minimum value of the image stack.

        Returns
        -------
        float
            The minimum value of the image stack.
        """
        return np.min(self.image_stack)
