import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.optimize import OptimizeResult, least_squares


def fit_ellipse_to_points(
    centers: np.ndarray,
) -> Tuple[int, int, int, int, int]:
    """Fit an ellipse to the points using least squares optimization.

    Parameters
    ----------
    centers : np.ndarray
        The centers of the largest blob in each image.

    Returns
    -------
    Tuple[int, int, int, int, int]
        The center of the ellipse (center_x, center_y), the semi-major
        axis (a), the semi-minor axis (b), and the rotation angle (theta).
    """
    # Convert centers to numpy array
    centers = np.array(centers)
    x = centers[:, 0]
    y = centers[:, 1]

    # Find the extreme points for the initial ellipse estimate
    topmost = centers[np.argmin(y)]
    rightmost = centers[np.argmax(x)]
    bottommost = centers[np.argmax(y)]
    leftmost = centers[np.argmin(x)]

    # Initial parameters: (center_x, center_y, semi_major_axis,
    # semi_minor_axis, rotation_angle)
    initial_center = np.mean(
        [topmost, bottommost, leftmost, rightmost], axis=0
    )
    semi_major_axis = np.linalg.norm(rightmost - leftmost) / 2
    semi_minor_axis = np.linalg.norm(topmost - bottommost) / 2
    rotation_angle = 0  # Start with no rotation

    initial_params = [
        initial_center[0],
        initial_center[1],
        semi_major_axis,
        semi_minor_axis,
        rotation_angle,
    ]

    logging.info("Fitting ellipse to points...")

    # Objective function to minimize: sum of squared distances to ellipse
    def ellipse_residuals(params, x, y):
        center_x, center_y, a, b, theta = params  # theta is in radians
        cos_angle = np.cos(theta)
        sin_angle = np.sin(theta)

        # Rotate the points to align with the ellipse axes
        x_rot = cos_angle * (x - center_x) + sin_angle * (y - center_y)
        y_rot = -sin_angle * (x - center_x) + cos_angle * (y - center_y)

        # Ellipse equation: (x_rot^2 / a^2) + (y_rot^2 / b^2) = 1
        return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1

    # Use least squares optimization to fit the ellipse to the points
    result: OptimizeResult = least_squares(
        ellipse_residuals,
        initial_params,
        args=(x, y),
        loss="huber",  # minimize the influence of outliers
    )

    # Extract optimized parameters
    center_x, center_y, a, b, theta = result.x

    return center_x, center_y, a, b, theta


def plot_ellipse_fit_and_centers(
    centers: np.ndarray,
    center_x: int,
    center_y: int,
    a: int,
    b: int,
    theta: int,
    image_stack: np.ndarray,
    debug_plots_folder: Path,
    saving_name: str = "ellipse_fit.png",
):
    """Plot the fitted ellipse on the largest blob centers.

    Parameters
    ----------
    centers : np.ndarray
        The centers of the largest blob in each image.
    center_x : int
        The x-coordinate of the center of the ellipse
    center_y : int
        The y-coordinate of the center of the ellipse
    a : int
        The semi-major axis of the ellipse
    b : int
        The semi-minor axis of the ellipse
    theta : int
        The rotation angle of the ellipse
    image_stack : np.ndarray
        The image stack to plot the ellipse on.
    debug_plots_folder : Path
        The folder to save the debug plot to.
    saving_name : str, optional
        The name of the file to save the plot to, by default "ellipse_fit.png".
    """
    # Convert centers to numpy array
    centers = np.array(centers)
    x = centers[:, 0]
    y = centers[:, 1]

    image_stack = np.asarray(image_stack)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    max_projection = image_stack.max(axis=0)
    #  plot behind a frame of the original image
    ax.imshow(max_projection, cmap="gray")
    ax.scatter(x, y, label="Largest Blob Centers", color="red")

    # Plot fitted ellipse
    ellipse = Ellipse(
        (center_x, center_y),
        width=2 * a,
        height=2 * b,
        angle=np.degrees(theta),
        edgecolor="blue",
        facecolor="none",
        label="Fitted Ellipse",
    )
    ax.add_patch(ellipse)

    # Plot center of fitted ellipse
    ax.scatter(
        center_x,
        center_y,
        color="green",
        marker="x",
        s=100,
        label="Ellipse Center",
    )

    # Add some plot formatting
    ax.set_xlim(0, image_stack.shape[1])
    ax.set_ylim(
        image_stack.shape[1], 0
    )  # Invert y-axis to match image coordinate system
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)
    ax.set_title("Fitted Ellipse on largest blob centers")
    ax.axis("off")

    #  save
    plt.savefig(debug_plots_folder / saving_name)