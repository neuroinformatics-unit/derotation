from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

data_folder = Path("/Users/lauraporta/local_data/rotation/230822_CAA_1120509/")
original_tiff_path = data_folder / "imaging/rotation_00001_ce.tif"
derotated_tiff_path = data_folder / "derotated_image_stack_full.tif"
derotated_csv_path = data_folder / "derotated_image_stack_full.csv"
frames_to_save_path = data_folder / "frames/"

original_tiff = tiff.imread(original_tiff_path)
derotated_tiff = tiff.imread(derotated_tiff_path)
df = pd.read_csv(derotated_csv_path)
time_subset = np.arange(435, 510)

df = df.iloc[time_subset]

angle = df["rotation_angle"].values


for i, t in enumerate(time_subset):
    ax1 = plt.subplot(221)
    ax1.set_title("Original image")
    ax1.imshow(original_tiff[t], cmap="gray")
    ax1.axis("off")

    ax2 = plt.subplot(222)
    ax2.set_title("Derotated image")
    ax2.imshow(derotated_tiff[t], cmap="gray")
    ax2.axis("off")

    ax3 = plt.subplot(212)
    ax3.set_title("Rotation angle")
    ax3.scatter(np.arange(0, len(angle)), angle, s=1)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Angle (degrees)")
    ax3.axvline(x=i, color="r", linestyle="--")

    plt.savefig(
        frames_to_save_path / f"frame_{i}.png", bbox_inches="tight", dpi=200
    )
    plt.close()
