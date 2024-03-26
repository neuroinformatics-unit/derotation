import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


def join_incremental_and_full_video(dataset_path, new_folder):
    incremental_video = tiff.imread(
        f"{dataset_path}/derotated_image_stack_incremental.tif"
    )
    full_video = tiff.imread(f"{dataset_path}/derotated_image_stack_full.tif")

    joined_video = np.concatenate((incremental_video, full_video), axis=0)

    tiff.imsave(
        f"{new_folder}/derotated_image_stack_full_and_incremental.tif",
        joined_video,
    )


def join_csv_files(dataset_path, new_folder):
    incremental_csv = pd.read_csv(
        f"{dataset_path}/derotated_image_stack_incremental.csv", delimiter=","
    )
    full_csv = pd.read_csv(
        f"{dataset_path}/derotated_image_stack_full.csv", delimiter=","
    )

    latest_frame = incremental_csv["frame"].iloc[-1]
    # update the frame number in the incremental csv
    full_csv["frame"] = full_csv["frame"] + latest_frame

    joined_csv = pd.concat([incremental_csv, full_csv], ignore_index=True)

    joined_csv.to_csv(
        f"{new_folder}/derotated_image_stack_full_and_incremental.csv",
        index=False,
    )


if __name__ == "__main__":
    dataset_path = sys.argv[1]
    new_folder = Path(dataset_path) / "merged"
    new_folder.mkdir(exist_ok=True)
    join_incremental_and_full_video(dataset_path, new_folder)
    join_csv_files(dataset_path, new_folder)
