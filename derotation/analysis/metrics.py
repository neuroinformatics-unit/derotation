from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import blob_log
from sklearn.cluster import DBSCAN


def stability_of_most_detected_blob(
    data,
    plot=True,
    blob_log_kwargs={
        "min_sigma": 7,
        "max_sigma": 10,
        "threshold": 0.95,
        "overlap": 0,
    },
    clip=True,
):
    mean_images_by_angle, debug_plots_folder = data

    #  clip all the images to the same contrast
    if clip:
        clipped_images = [
            np.clip(img, np.percentile(img, 99), np.percentile(img, 99.99))
            for img in mean_images_by_angle
        ]
    else:
        clipped_images = mean_images_by_angle

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

        # save
        plt.savefig(debug_plots_folder / "detected_blobs.png")
        plt.close()

    # Flatten the blob list and add frame indices
    all_blobs = []
    for frame_idx, frame_blobs in enumerate(blobs):
        for blob in frame_blobs:
            all_blobs.append([*blob, frame_idx])
    all_blobs = np.array(all_blobs)

    # Use DBSCAN to cluster blobs based on proximity
    coords = all_blobs[:, :3]  # x, y, radius
    clustering = DBSCAN(eps=10, min_samples=2).fit(
        coords
    )  # Adjust eps as needed
    all_blobs = np.column_stack(
        (all_blobs, clustering.labels_)
    )  # Add cluster labels

    cluster_counts = Counter(all_blobs[:, -1])  # Cluster labels
    most_detected_label = max(cluster_counts, key=cluster_counts.get)

    # Extract blobs belonging to the most detected cluster
    most_detected_blobs = all_blobs[all_blobs[:, -1] == most_detected_label]

    #  Calculate range (peak to peak)
    ptp = np.ptp(most_detected_blobs[:, 0]) + np.ptp(most_detected_blobs[:, 1])
    #  Calculate stability (standard deviation)
    std = np.sqrt(
        np.var(most_detected_blobs[:, 0]) + np.var(most_detected_blobs[:, 1])
    )

    #  plot the most detected blobs centers
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(clipped_images[0], cmap="gray")
        for blob in most_detected_blobs:
            y, x, *_ = blob
            # plot an x on the center
            plt.scatter(x, y, color="red", marker="x")

        # save
        plt.savefig(debug_plots_folder / "most_detected_blob_centers.png")
        plt.close()

    return ptp, std
