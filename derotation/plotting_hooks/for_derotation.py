"""
This module contains hooks for making plots while the derotation is running.
The hooks are called at specific points in the derotation process, specifically
when a line is added to the derotated image and when a frame is completed.
Also, a maximum projection plot is generated at the end of the first rotation.
They are useful for debugging purposes and for generating figures.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def line_addition(
    derotated_filled_image: np.ndarray,
    rotated_line: np.ndarray,
    image_counter: int,
    line_counter: int,
    angle: float,
    original_image: np.ndarray,
):
    """
    Hook for plotting the derotated image and the current rotated line.

    Parameters
    ----------
    derotated_filled_image : np.ndarray
        The derotated image.
    rotated_line : np.ndarray
        The rotated line.
    image_counter : int
        The current frame number.
    line_counter : int
        The current line number.
    angle : float
        The rotation angle of the line
    original_image : np.ndarray
        The original image from which the line was taken.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #  background fig color: black
    fig.patch.set_facecolor("black")

    ax[0].imshow(original_image, cmap="viridis")
    #  highlight the line in the original image
    ax[0].plot(
        [0, original_image.shape[1] - 1],
        [line_counter, line_counter],
        color="red",
        linewidth=2,
    )
    ax[0].set_title(
        f"Take line {line_counter}\nfrom original image,\n then rotate it of "
        f"{angle:.2f} degrees"
    )
    ax[0].title.set_color("white")
    ax[0].axis("off")

    ax[1].imshow(derotated_filled_image, cmap="viridis")
    ax[1].set_title(
        f"Place the line in a new image\nto build frame {image_counter}"
    )
    ax[1].title.set_color("white")
    ax[1].axis("off")

    #  plot on top axis 1 the rotated_line with a red colormap
    ax[1].imshow(rotated_line, cmap="Reds", alpha=0.5)

    Path("debug/lines/").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"debug/lines/derotated_image_{image_counter}_line_{line_counter}.png",
        dpi=300,
    )
    plt.close()


def image_completed(
    derotated_image_stack: np.ndarray,
    image_counter: int,
    frame_of_interest: int = 149,
):
    """Hook for plotting the maximum projection of the derotated image stack
    after the first rotation and the current derotated image.

    Parameters
    ----------
    derotated_image_stack : np.ndarray
        The derotated image stack.
    image_counter : int
        The current frame number.
    frame_of_interest : int, optional
        A frame number for which the maximum projection will be saved
        by cumulating all the frames up to that frame.
        Suggestion: set it to the last frame of the first rotation.
    """
    if image_counter == frame_of_interest:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(
            np.max(derotated_image_stack[:image_counter], axis=0),
            cmap="viridis",
        )
        ax.axis("off")
        Path("debug/").mkdir(parents=True, exist_ok=True)
        plt.savefig("debug/max_projection.png", dpi=300)
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(derotated_image_stack[image_counter], cmap="viridis")
    ax.axis("off")
    Path("debug/frames/").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"debug/frames/derotated_image_{image_counter}.png", dpi=300)
    plt.close()
