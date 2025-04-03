"""This module contains functions to calculate metrics for the derotation
analysis. The ``ptd_of_most_detected_blob`` function calculates the peak to
peak distance of the centers of the most detected blob in the derotated stack
across all frames. The function uses the ``blob_log`` function from the
``skimage.feature`` module to detect the blobs and the DBSCAN algorithm from
the ``sklearn.cluster`` module to cluster the blobs based on proximity. The
function returns the peak to peak distance of the centers of the most detected
blob."""

from collections import Counter
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_log
from sklearn.cluster import DBSCAN


def ptd_of_most_detected_blob(
    mean_images_by_angle: np.ndarray,
    plot: bool = True,
    blob_log_kwargs: dict = {
        "min_sigma": 7,
        "max_sigma": 10,
        "threshold": 0.95,
        "overlap": 0,
    },
    debug_plots_folder: Path = Path("/debug_plots"),
    image_names: List[str] = [
        "detected_blobs.png",
        "most_detected_blob_centers.png",
    ],
    DBSCAN_max_distance: float = 10.0,
    clipping_percentiles: List[float] = [99.0, 99.99],
) -> float:
    """Calculate the peak to peak distance of the centers of the most
    detected blob in the derotated stack across all frames.

    Parameters
    ----------
    mean_images_by_angle : np.ndarray
        The derotated stack of images.
    plot : bool, optional
        Whether to plot the detected blobs, by default True
    blob_log_kwargs : _type_, optional
        The parameters for the blob detection algorithm, by default
        { "min_sigma": 7, "max_sigma": 10, "threshold": 0.95, "overlap": 0, }
    debug_plots_folder : str, optional
        The folder to save the debugging plots, by default "/debug_plots"
    image_names : List[str], optional
       The names of the images to save if plot is True, by default
       ["detected_blobs.png", "most_detected_blob_centers.png"]
    DBSCAN_max_distance : float, optional
        The maximum distance between two samples for one to be considered as
        in the neighborhood of the other, by default 10.0
    clipping_percentiles : List[float], optional
        The percentiles to clip the images to, by default [99.0, 99.99]

    Returns
    -------
    float
        The peak to peak distance of the centers of the most detected blob.
    """
    #  clip all the images to the same contrast
    clipped_images = [
        np.clip(
            img,
            np.percentile(img, clipping_percentiles[0]),
            np.percentile(img, clipping_percentiles[1]),
        )
        for img in mean_images_by_angle
    ]

    # Detect the blobs in the derotated stack in each frame
    # blobs is a list(list(x, y, sigma)) of the detected blobs for every frame
    blobs = [
        blob_log(
            img,
            min_sigma=blob_log_kwargs["min_sigma"],
            max_sigma=blob_log_kwargs["max_sigma"],
            threshold=blob_log_kwargs["threshold"],
            overlap=blob_log_kwargs["overlap"],
        )
        for img in clipped_images
    ]

    # plot image with center of blobs
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(clipped_images[0], cmap="gray")
        for blob in blobs[0]:
            y, x, r = blob
            c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
            plt.gca().add_artist(c)

        ax.axis("off")

        # save
        plt.savefig(debug_plots_folder / image_names[0])
        plt.close()

    # Flatten the blob list and add frame indices
    _blobs = []
    for frame_idx, frame_blobs in enumerate(blobs):
        for blob in frame_blobs:
            _blobs.append([*blob, frame_idx])
    all_blobs = np.array(_blobs)

    # Use DBSCAN to cluster blobs based on proximity

    coords = all_blobs[:, :3]  # x, y, radius
    DBSCAN_max_distance = float(DBSCAN_max_distance)
    clustering = DBSCAN(
        eps=DBSCAN_max_distance,
        min_samples=2,
    ).fit(coords)
    all_blobs = np.column_stack(
        (all_blobs, clustering.labels_)
    )  # Add cluster labels

    cluster_counts = Counter(all_blobs[:, -1])  # Cluster labels
    most_detected_label = max(cluster_counts, key=lambda k: cluster_counts[k])

    # Extract blobs belonging to the most detected cluster
    most_detected_blobs = all_blobs[all_blobs[:, -1] == most_detected_label]

    #  Calculate range (peak to peak)
    ptp = np.ptp(most_detected_blobs[:, 0]) + np.ptp(most_detected_blobs[:, 1])

    #  plot the most detected blobs centers
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(clipped_images[0], cmap="gray")
        for blob in most_detected_blobs:
            y, x, *_ = blob
            # plot an x on the center
            plt.scatter(x, y, color="red", marker="x")

        ax.axis("off")

        # save
        plt.savefig(debug_plots_folder / image_names[1])
        plt.close()

    return ptp
