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
):
    """
    Hook for plotting the derotated image and the current rotated line.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(derotated_filled_image, cmap="viridis")
    ax[0].set_title(f"Frame {image_counter}")
    ax[0].axis("off")

    ax[1].imshow(rotated_line, cmap="viridis")
    ax[1].set_title(f"Line {line_counter}, angle: {angle:.2f}")
    ax[1].axis("off")

    Path("debug/lines/").mkdir(parents=True, exist_ok=True)
    plt.savefig(
        f"debug/lines/derotated_image_{image_counter}_line_{line_counter}.png",
        dpi=300,
    )
    plt.close()


def image_completed(derotated_image_stack, image_counter):
    """Hook for plotting the maximum projection of the derotated image stack
    after the first rotation (which ends ~149th frame) and the current
    derotated image.
    """
    if image_counter == 149:
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