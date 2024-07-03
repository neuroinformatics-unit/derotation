import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff


def make_mean_image_every_10deg(dataset_path):
    full_video = tiff.imread(
        f"{dataset_path}/derotated_image_stack_full_and_incremental.tif"
    )
    full_csv = pd.read_csv(
        f"{dataset_path}/derotated_image_stack_full_and_incremental.csv",
        delimiter=",",
    )

    mean_images = []
    #  images around 5, 10, 15, 20, etc. degrees
    tollerace_deg = 1
    for angle in range(0, 360, 10):
        images = full_video[
            np.where(
                np.abs(full_csv["rotation_angle"] - angle) < tollerace_deg
            )
        ]
        mean_image = np.mean(images, axis=0)
        mean_images.append(mean_image)

    mean_images = np.array(mean_images)

    path_for_mean_images = Path(dataset_path) / "mean_images"
    Path(path_for_mean_images).mkdir(exist_ok=True)
    for i, mean_image in enumerate(mean_images):
        angle = i * 10
        plt.imsave(
            f"{path_for_mean_images}/mean_image_{angle}.png",
            mean_image,
            cmap="gray",
        )


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    make_mean_image_every_10deg(dataset_path)
