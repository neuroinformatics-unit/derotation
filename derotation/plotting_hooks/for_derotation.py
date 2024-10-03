from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def line_addition(
    derotated_filled_image, rotated_line, image_counter, line_counter, angle
):
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
    """Hook for plotting the image stack after derotation.
    It is useful for debugging purposes.
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
