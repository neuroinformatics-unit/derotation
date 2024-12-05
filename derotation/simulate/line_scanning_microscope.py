from typing import Optional, Tuple

import numpy as np
import tqdm
from scipy.ndimage import affine_transform


class Rotator:
    def __init__(
        self,
        angles: np.ndarray,
        image_stack: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        rotation_plane_angle: Optional[float] = None,
        rotation_plane_orientation: Optional[float] = None,
        blank_pixel_val: Optional[float] = None,
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

        if rotation_plane_orientation is not None:
            assert rotation_plane_angle is not None, (
                "If rotation_plane_orientation is provided, "
                + "rotation_plane_angle should be provided as well."
            )

        self.image_stack = image_stack
        self.num_lines_per_frame = image_stack.shape[1]

        if center is None:
            self.center = np.array(image_stack.shape[1:]) / 2
        else:
            self.center = np.array(center)

        if rotation_plane_angle is None:
            self.rotation_plane_angle: float = 0
            self.ps: int = 0
            self.image_size = image_stack.shape[1]
        else:
            self.rotation_plane_angle = rotation_plane_angle
            if rotation_plane_orientation is not None:
                self.rotation_plane_orientation = rotation_plane_orientation
            else:
                self.rotation_plane_orientation = 0

            self.create_homography_matrices()
            print(f"Pixel shift: {self.ps}")
            print(f"New image size: {self.image_size}")

        #  reshape the angles to the shape of the image
        #  stack to ease indexing
        self.angles = angles[: image_stack.shape[0] * self.image_size].reshape(
            image_stack.shape[0], self.image_size
        )
        print("New angles shape:", self.angles.shape)

        if blank_pixel_val is None:
            self.blank_pixel_val = self.get_blank_pixels_value()
        else:
            self.blank_pixel_val = blank_pixel_val

    def create_homography_matrices(self) -> None:
        #  from the scanning plane to the rotation plane
        self.homography_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(np.radians(self.rotation_plane_angle)), 0],
                [0, 0, 1],
            ]
        )

        #  from the rotation plane to the scanning plane
        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)

        #  store pixels shift based on inverse homography
        line_length = self.image_stack.shape[2]
        self.ps = (
            line_length
            - np.round(
                np.abs(
                    line_length * np.cos(np.radians(self.rotation_plane_angle))
                )
            ).astype(int)
            + 1
        )

        #  round to the nearest even number
        if self.ps % 2 != 0:
            self.ps -= 1

        #  final image size depends on the pixel shift
        self.image_size = line_length - self.ps

    def crop_image(self, image: np.ndarray) -> np.ndarray:
        if self.ps == 0:
            return image
        else:
            return image[
                # centered in the rows
                self.ps // 2 : -self.ps // 2,
                # take the left side of the image
                : self.image_size,
            ]

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
        rotated_image_stack = np.empty(
            (self.image_stack.shape[0], self.image_size, self.image_size),
            dtype=np.float64,
        )

        for i, image in tqdm.tqdm(
            enumerate(self.image_stack), total=self.image_stack.shape[0]
        ):
            is_this_frame_rotating = not np.all(
                # don't bother if rotation is less than 0.01 degrees
                np.isclose(self.angles[i], 0, atol=1e-2)
            )
            if is_this_frame_rotating:
                for j, angle in enumerate(self.angles[i]):
                    if angle == 0:
                        rotated_image_stack[i][j] = self.crop_image(image)[j]
                    else:
                        # rotate the whole image by the angle
                        rotated_image = self.rotate_sample(image, angle)

                        # if the rotation plane angle is not 0,
                        # apply the homography
                        if self.rotation_plane_angle != 0:
                            rotated_image = self.apply_homography(
                                rotated_image, "rotation_to_scanning_plane"
                            )
                            rotated_image = self.crop_image(rotated_image)

                        # store the rotated image line
                        rotated_image_stack[i][j] = rotated_image[j]
            else:
                rotated_image_stack[i] = self.crop_image(image)

        return rotated_image_stack

    def apply_homography(
        self, image: np.ndarray, direction: str
    ) -> np.ndarray:
        if direction == "scanning_to_rotation_plane":
            # backward transformation
            return affine_transform(
                image,
                self.homography_matrix,
                offset=self.center - self.center,
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=self.get_blank_pixels_value(),
            )
        elif direction == "rotation_to_scanning_plane":
            # forward transformation
            image = affine_transform(
                image,
                self.inverse_homography_matrix,
                offset=self.center - self.center,
                output_shape=image.shape,
                order=0,
                mode="constant",
                cval=self.get_blank_pixels_value(),
            )

            #  rotate the image back to the scanning plane angle
            if self.rotation_plane_orientation != 0:
                image = self.rotate_sample(
                    image, self.rotation_plane_orientation
                )

            return image

    def rotate_sample(self, image: np.ndarray, angle: float) -> np.ndarray:
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
        offset = self.center - rotation_matrix @ self.center

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
