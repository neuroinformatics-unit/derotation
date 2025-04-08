"""
This module contains the ``SyntheticData`` class that generates synthetic data
for testing and developing the derotation pipeline.
"""

import copy
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from derotation.analysis.fit_ellipse import derive_angles_from_ellipse_fits
from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)
from derotation.derotate_by_line import derotate_an_image_array_line_by_line
from derotation.simulate.line_scanning_microscope import Rotator


class SyntheticData:
    """Class to generate synthetic data for testing and developing the
    derotation pipeline.

    The synthetic data consists of:
        - a 2D image with two circles, one bright and one dim (optional),
          by default in the top center and bottom right, respectively. Use
          the ``center_of_bright_cell`` and ``center_of_dimmer_cell``
          parameters to change the location of the circles. If padding and
          background value are provided, the image will be padded and the
          background value will be set to the background_value.
          Padding helps increasing the field of view in order to
          accommodate for larger rotations and have more data to derotate.
          The background value is useful to distinguish pixels from the
          original image from those that are never evaluated, which will
          remain 0. NB: padding will alter the number of lines per frame.
        - two angle arrays for incremental and sinusoidal rotation
        - one 3D image stack with the 2D image repeated for a given number
          of frames (this is going to be the input for the ``Rotator``)
        - two rotated movies made with incremental and sinusoidal rotations
        - two derotated movies:
            - one made with a mock of the ``IncrementalPipeline``, used then
              to estimate the center of rotation and the ellipse fits
            - one made just with ``derotate_an_image_array_line_by_line`` with
              the sinusoidal rotation angles

    Why mocking the ``IncrementalPipeline``?
    The ``IncrementalPipeline`` has the responsibility to find the center of
    rotation but with mock data we cannot use it off the shelf because it
    is too bound to signals coming from a real motor in the
    ``calculate_mean_images`` method and in the constructor.

    See the integration_pipeline method for the full pipeline.

    Parameters
    ----------
    center_of_bright_cell : tuple, optional
        The location of the brightest cell, by default (50, 10)
    center_of_dimmer_cell : tuple, optional
        The location of the dimmer cell, by default (60, 60)
    pad : int, optional
        Padding around the image, by default 0
    background_value : int, optional
        Background value, by default 80
    lines_per_frame : int, optional
        Number of lines per frame, by default 100
    second_cell : bool, optional
        Add an extra dimmer cell, by default True
    radius : int, optional
        Radius of the circles, by default 5
    num_frames : int, optional
        Number of frames in the 3D image stack, by default 100
    center_of_rotation_offset : tuple, optional
        The offset of the center of rotation, by default (0, 0)
    rotation_plane_angle : int, optional
        The angle of the rotation plane, by default 0
    rotation_plane_orientation : int, optional
        The orientation of the rotation plane, by default 0
    plots : bool, optional
        Whether to plot debugging plots, by default False
    """

    def __init__(
        self,
        center_of_bright_cell: Tuple[int, int] = (50, 10),
        center_of_dimmer_cell: Tuple[int, int] = (60, 60),
        pad: int = 0,
        background_value: int = 80,
        lines_per_frame: int = 100,
        second_cell: bool = True,
        radius: int = 5,
        num_frames: int = 100,
        center_of_rotation_offset: Tuple[int, int] = (0, 0),
        rotation_plane_angle: int = 0,
        rotation_plane_orientation: int = 0,
        plots: bool = False,
    ):
        """Initialize the ``SyntheticData`` object."""

        self.center_of_bright_cell = center_of_bright_cell
        self.center_of_dimmer_cell = center_of_dimmer_cell
        self.lines_per_frame = lines_per_frame
        self.second_cell = second_cell
        self.radius = radius
        self.num_frames = num_frames
        self.center_of_rotation_offset = center_of_rotation_offset
        self.rotation_plane_angle = rotation_plane_angle
        self.rotation_plane_orientation = rotation_plane_orientation
        self.plots = plots
        self.pad = pad
        self.background_value = background_value

    def integration_pipeline(
        self,
    ) -> np.ndarray:
        """Integration pipeline that combines the incremental and sinusoidal
        rotation pipelines to derotate a 3D image stack.

        The pipeline rotates the image stack incrementally and sinusoidally
        and then derotates the sinusoidal stack using the center of rotation
        estimated by the incremental pipeline.

        The pipeline also plots debugging plots if the ``plots`` parameter is
        set to `True`.
        """

        # -----------------------------------------------------
        # Create sample image with two cells
        self.image = self.create_sample_image_with_cells()

        # Create the 3D image stack
        image_stack = self.create_image_stack()

        # Generate rotation angles
        incremental_angles, sinusoidal_angles = self.create_rotation_angles(
            image_stack.shape
        )

        #  -----------------------------------------------------
        # Initialize Rotator for incremental rotation
        rotator_incremental = Rotator(
            incremental_angles,
            image_stack,
            center_offset=self.center_of_rotation_offset,
            rotation_plane_angle=self.rotation_plane_angle,
            rotation_plane_orientation=self.rotation_plane_orientation,
        )
        # Rotate the image stack incrementally
        rotated_stack_incremental = rotator_incremental.rotate_by_line()

        # Initialize Rotator for sinusoidal rotation
        rotator_sinusoidal = Rotator(
            sinusoidal_angles,
            image_stack,
            center_offset=self.center_of_rotation_offset,
            rotation_plane_angle=self.rotation_plane_angle,
            rotation_plane_orientation=self.rotation_plane_orientation,
        )
        # Rotate the image stack sinusoidally
        self.rotated_stack_sinusoidal = rotator_sinusoidal.rotate_by_line()

        #  -----------------------------------------------------
        # Derotate the sinusoidal stack using the center of rotation
        # estimated by the incremental pipeline

        # Get the center of rotation with a mock of the IncrementalPipeline
        center_of_rotation, ellipse_fits = self.get_center_of_rotation(
            rotated_stack_incremental, incremental_angles
        )

        self.center = rotator_sinusoidal.post_homography_center
        self.fitted_center = center_of_rotation

        # derive orientation and angle from the ellipse fits
        if self.rotation_plane_angle != 0:
            theta, orientation = derive_angles_from_ellipse_fits(ellipse_fits)
            self.rotation_plane_angle = np.round(theta, 1)
            self.rotation_plane_orientation = np.round(orientation, 1)

        # Derotate the sinusoidal stack
        self.derotated_sinusoidal = derotate_an_image_array_line_by_line(
            self.rotated_stack_sinusoidal,
            sinusoidal_angles,
            center=self.fitted_center,
            use_homography=self.rotation_plane_angle != 0,
            rotation_plane_angle=self.rotation_plane_angle,
            rotation_plane_orientation=self.rotation_plane_orientation,
        )

        #  -----------------------------------------------------
        #  Debugging plots
        #  Will be run if the script is run as a standalone script
        if self.plots:
            self.plot_original_image()
            self.plot_each_frame()
            self.plot_mean_projection()
            self.plot_angles(incremental_angles, sinusoidal_angles)
            self.plot_a_few_rotated_frames(
                rotated_stack_incremental, self.rotated_stack_sinusoidal
            )
            self.plot_derotated_frames(self.derotated_sinusoidal)

    #  -----------------------------------------------------
    #  Prepare the 3D image stack and the rotation angles
    #  -----------------------------------------------------

    def create_sample_image_with_cells(self) -> np.ndarray:
        """Create a 2D image with two circles, one bright and one dim
        (optional) by default in the top center and bottom right,
        respectively.

        Location of the circles can be changed by providing the
        ``center_of_bright_cell`` and ``center_of_dimmer_cell`` parameters.

        If specified, the image will be padded and the background value
        will be set to the background_value.

        Parameters
        ----------
        center_of_bright_cell : Tuple[int, int], optional
            Location of brightest cell, by default (50, 10)
        center_of_dimmer_cell : Tuple[int, int], optional
            Location of dimmer cell, by default (60, 60)
        lines_per_frame : int, optional
            Number of lines per frame, by default 100
        second_cell : bool, optional
            Add an extra dimmer cell, by default True
        radius : int, optional
            Radius of the circles, by default 5

        Returns
        -------
        np.ndarray
            2D image with two circles, one bright and one dim
        """

        def make_circle(image, center, intensity):
            y, x = np.ogrid[: image.shape[0], : image.shape[1]]
            mask = (x - center[0]) ** 2 + (
                y - center[1]
            ) ** 2 <= self.radius**2
            image[mask] = intensity

            return image

        # Initialize a black image of size 100x100
        image = np.zeros(
            (self.lines_per_frame, self.lines_per_frame), dtype=np.uint8
        )

        # Draw a white circle in the top center
        image = make_circle(image, self.center_of_bright_cell, 255)

        if self.second_cell:
            #  add an extra gray circle at the bottom right
            image = make_circle(image, self.center_of_dimmer_cell, 128)

        if self.pad > 0:
            image = np.pad(
                image,
                ((self.pad, self.pad), (self.pad, self.pad)),
                mode="constant",
            )
            #  adjust the center of the bright cell
            self.center_of_bright_cell = (
                self.center_of_bright_cell[0] + self.pad,
                self.center_of_bright_cell[1] + self.pad,
            )
            #  adjust the center of the dimmer cell
            self.center_of_dimmer_cell = (
                self.center_of_dimmer_cell[0] + self.pad,
                self.center_of_dimmer_cell[1] + self.pad,
            )
            self.lines_per_frame = image.shape[0]

        if self.background_value != 0:
            image[image == 0] = self.background_value

        return image

    def create_image_stack(self) -> np.ndarray:
        """Create a 3D image stack by repeating the 2D image
        for a given number of frames.

        Returns
        -------
        np.ndarray
            3D image stack
        """
        return np.array([self.image for _ in range(self.num_frames)])

    @staticmethod
    def create_rotation_angles(
        image_stack_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create rotation angles for incremental and sinusoidal rotation
        for a given 3D image stack.

        Parameters
        ----------
        image_stack_shape : Tuple[int, int]
            Shape of the 3D image stack

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of incremental and sinusoidal rotation angles
        """

        # Generate rotation angles for incremental rotation
        # which consists of 36 steps of 10 degrees each
        # If plotted they look like a staircase
        num_lines_total = image_stack_shape[1] * image_stack_shape[0]
        num_steps = 360 // 10
        incremental_angles = np.arange(0, num_steps * 10, 10)
        incremental_angles = np.repeat(
            incremental_angles, num_lines_total // len(incremental_angles)
        )

        if len(incremental_angles) < num_lines_total:
            incremental_angles = np.concatenate(
                (
                    incremental_angles,
                    [incremental_angles[-1]]
                    * (num_lines_total - len(incremental_angles)),
                )
            )

        # Generate rotation angles for sinusoidal rotation
        max_rotation = 360  # max rotation angle
        num_cycles = 1
        sinusoidal_angles = max_rotation * np.sin(
            np.linspace(0, num_cycles * 2 * np.pi, num_lines_total)
        )

        return incremental_angles.astype("float64"), sinusoidal_angles

    #  -----------------------------------------------------
    #  Integration pipeline with mock of the IncrementalPipeline
    #  -----------------------------------------------------

    def get_center_of_rotation(
        self,
        rotated_stack_incremental: np.ndarray,
        incremental_angles: np.ndarray,
    ) -> Tuple[Tuple[int, int], Any]:
        """Get the center of rotation by using the ``IncrementalPipeline``.

        The Incremental pipeline has the responsibility to find the center of
        rotation but with mock data we cannot use it off the shelf because it
        is too bound to signals coming from a real motor in the
        ``calculate_mean_images`` method and in the constructor.
        We will create a mock class that inherits from the
        ``IncrementalPipeline`` and overwrite the ``calculate_mean_images``
        method to work with our mock data.

        Parameters
        ----------
        rotated_stack_incremental : np.ndarray
            The 3D image stack rotated incrementally
        incremental_angles : np.ndarray
            The rotation angles for incremental rotation

        Returns
        -------
        Tuple[int, int]
            The center of rotation
        """

        make_plots = self.plots

        # Mock class to use the IncrementalPipeline
        class MockIncrementalPipeline(IncrementalPipeline):
            def __init__(self):
                # Overwrite the constructor and provide the mock data
                self.image_stack = rotated_stack_incremental
                self.rot_deg_frame = incremental_angles[
                    :: rotated_stack_incremental.shape[1]
                ][: rotated_stack_incremental.shape[0]]
                self.num_frames = rotated_stack_incremental.shape[0]
                self.num_lines_per_frame = rotated_stack_incremental.shape[1]

                self.debugging_plots = make_plots
                self.debug_plots_folder = Path("debug/")

            @staticmethod
            def calculate_mean_images(
                image_stack: np.ndarray,
                rot_deg_frame: np.ndarray,
                round_decimals: int = 0,
            ) -> list:
                #  Override original method as it is too bound
                #  to signal coming from a real motor
                angles_subset = copy.deepcopy(rot_deg_frame)
                rounded_angles = np.round(angles_subset, round_decimals)

                mean_images = []
                for i in np.arange(10, 360, 10):
                    images = image_stack[rounded_angles == i]
                    mean_image = np.mean(images, axis=0)

                    mean_images.append(mean_image)

                return mean_images

        # Use the mock class to find the center of rotation
        pipeline = MockIncrementalPipeline()
        center_of_rotation = pipeline.find_center_of_rotation()
        return center_of_rotation, pipeline.all_ellipse_fits

    # -----------------------------------------------------
    # Debugging plots
    # -----------------------------------------------------

    def plot_angles(
        self, incremental_angles: np.ndarray, sinusoidal_angles: np.ndarray
    ):
        """Plot the incremental and sinusoidal rotation angles.

        Parameters
        ----------
        incremental_angles : np.ndarray
            Incremental rotation angles
        sinusoidal_angles : np.ndarray
            Sinusoidal rotation angles
        """

        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
        fig.suptitle("Rotation Angles")

        axs[0].plot(incremental_angles, label="Incremental Rotation")
        axs[0].set_title("Incremental Rotation Angles")
        axs[0].set_ylabel("Angle (degrees)")
        axs[0].set_xlabel("Line Number")
        axs[0].legend()

        axs[1].plot(sinusoidal_angles, label="Sinusoidal Rotation")
        axs[1].set_title("Sinusoidal Rotation Angles")
        axs[1].set_ylabel("Angle (degrees)")
        axs[1].set_xlabel("Line Number")
        axs[1].legend()

        plt.tight_layout()

        plt.savefig(
            f"debug/rotation_angles{self.center_of_rotation_offset}_{self.rotation_plane_angle}_{self.rotation_plane_orientation}.png"
        )

        plt.close()

    def plot_a_few_rotated_frames(
        self,
        rotated_stack_incremental: np.ndarray,
        rotated_stack_sinusoidal: np.ndarray,
    ):
        """Plot a few frames from the rotated stacks.

        Parameters
        ----------
        rotated_stack_incremental : np.ndarray
            The 3D image stack rotated incrementally
        rotated_stack_sinusoidal : np.ndarray
            The 3D image stack rotated sinusoidally
        """

        fig, axs = plt.subplots(2, 5, figsize=(15, 6))

        for i, ax in enumerate(axs[0]):
            ax.imshow(rotated_stack_incremental[i * 5], cmap="gray")
            ax.set_title(f"Frame {i * 5}")
            ax.axis("off")

        for i, ax in enumerate(axs[1]):
            ax.imshow(rotated_stack_sinusoidal[i * 5], cmap="gray")
            ax.set_title(f"Frame {i * 5}")
            ax.axis("off")

        plt.savefig(
            f"debug/rotated_stacks{self.center_of_rotation_offset}_{self.rotation_plane_angle}_{self.rotation_plane_orientation}.png"
        )

        plt.close()

    def plot_derotated_frames(self, derotated_sinusoidal: np.ndarray):
        """Plot a few frames from the derotated stack.

        Parameters
        ----------
        derotated_sinusoidal : np.ndarray
            The 3D image stack derotated sinusoidally
        """

        fig, axs = plt.subplots(2, 5, figsize=(15, 6))

        for i, ax in enumerate(axs[0]):
            ax.imshow(derotated_sinusoidal[i * 5], cmap="gray")
            ax.set_title(f"Frame {i * 5}")
            ax.axis("off")

        for i, ax in enumerate(axs[1]):
            ax.imshow(derotated_sinusoidal[i * 5 + 1], cmap="gray")
            ax.set_title(f"Frame {i * 5 + 1}")
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            f"debug/derotated_sinusoidal{self.center_of_rotation_offset}_{self.rotation_plane_angle}_{self.rotation_plane_orientation}.png"
        )

        plt.close()

    def plot_original_image(self):
        """
        Plot the original image with the two circles.
        """

        fig, ax = plt.subplots()
        fig.patch.set_facecolor("black")
        ax.imshow(self.image, cmap="gray", vmin=0, vmax=255)
        plt.savefig(
            f"debug/image_{self.center_of_rotation_offset}_{self.rotation_plane_angle}_{self.rotation_plane_orientation}.png"
        )
        plt.close()

    def plot_each_frame(self):
        """Plot each frame of the rotated and derotated stacks as an image."""
        Path("debug/frames_rotator_orig/").mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self.rotated_stack_sinusoidal):
            plt.close()
            fig, ax = plt.subplots()
            #  background black
            fig.patch.set_facecolor("black")
            ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
            plt.savefig(
                f"debug/frames_rotator_orig/rotated_image_{i}_{self.center_of_rotation_offset}_{self.rotation_plane_angle}_{self.rotation_plane_orientation}.png"
            )
            plt.close()

        Path("debug/frames_derotator/").mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(self.derotated_sinusoidal):
            fig, ax = plt.subplots()
            #  background black
            fig.patch.set_facecolor("black")
            ax.imshow(frame, cmap="gray")
            plt.savefig(
                f"debug/frames_derotator/derotated_image_{i}_{self.center_of_rotation_offset}_{self.rotation_plane_angle}_{self.rotation_plane_orientation}.png"
            )
            plt.close()

    def plot_mean_projection(self):
        """Plot the mean projection of the derotated sinusoidal stack."""
        mean_projection = np.mean(self.derotated_sinusoidal, axis=0)
        fig, ax = plt.subplots()
        ax.imshow(mean_projection, cmap="gray")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"debug/mean_projection_{self.center_of_rotation_offset}_{self.rotation_plane_angle}_{self.rotation_plane_orientation}.png"
        )
        plt.close()
