from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.io import loadmat
from tifffile import imsave

from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline()

pipeline.process_analog_signals()

# import already rotated images
path_tif = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/derotated/masked.tif"
)
derotated = tiff.imread(path_tif)

# rotation speed
path_randperm = Path(
    "/Users/lauraporta/local_data/rotation/stimlus_randperm.mat"
)
pseudo_random = loadmat(path_randperm)
rotation_speed = pseudo_random["stimulus_random"][:, 0]

full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
direction = np.where(
    full_rotation_blocks_direction, -1, 1
)  # 1 is counterclockwise, -1 is clockwise

# in rotation ticks time
frame_start, frame_end, _ = pipeline.get_starting_and_ending_times(
    clock_type="frame"
)

rot_start = pipeline.rot_blocks_idx["start"]
rot_end = pipeline.rot_blocks_idx["end"]
all_initial_frames_counted = False
_rotated_frames = []
rotation_index = 0
for f_idx, f_start in enumerate(frame_start):
    try:
        f_end = frame_start[f_idx + 1]
    except IndexError:
        break
    no_rotation_start = (f_start < rot_start[0]) and (f_end < rot_start[0])

    if no_rotation_start:
        row = {
            "frame_idx": f_idx,
            "is_rotating": False,
            "rotation_idx": "no_rotation",
            "rotation_speed": np.nan,
            "direction": np.nan,
            "starts_in_frame": False,
        }
        _rotated_frames.append(row)
    elif rotation_index < len(rot_start):
        start_rotation = rot_start[rotation_index]
        try:
            next_rotation_starts = rot_start[rotation_index + 1]
        except IndexError:
            next_rotation_starts = rot_end[rotation_index]
        rotation_starts_in_frame = (start_rotation > f_start) and (
            start_rotation < f_end
        )
        rotation_already_started = (start_rotation < f_start) and (
            next_rotation_starts > f_start
        )
        # rotation_ends_in_frame = (end_rotation >= f_start) and (
        #     end_rotation <= f_end
        # )
        try:
            next_rotation_starts_in_next_frame = (
                rot_start[rotation_index + 1] > frame_start[f_idx + 1]
            ) and (rot_start[rotation_index + 1] < frame_start[f_idx + 2])
        except IndexError:
            break

        if rotation_starts_in_frame:
            row = {
                "frame_idx": f_idx,
                "is_rotating": True,
                "rotation_idx": rotation_index,
                "rotation_speed": rotation_speed[rotation_index],
                "direction": direction[rotation_index],
                "starts_in_frame": True,
            }
            _rotated_frames.append(row)
        elif rotation_already_started:
            row = {
                "frame_idx": f_idx,
                "is_rotating": True,
                "rotation_idx": rotation_index,
                "rotation_speed": rotation_speed[rotation_index],
                "direction": direction[rotation_index],
                "starts_in_frame": False,
            }
            _rotated_frames.append(row)
        # elif not next_rotation_starts_in_frame:
        #     # we are in a rotation interval but not in a rotation frame
        #     row = {
        #         "frame_idx": f_idx,
        #         "is_rotating": False,
        #         "rotation_idx": rotation_index,
        #         "rotation_speed": rotation_speed[rotation_index],
        #         "direction": direction[rotation_index],
        #         "starts_in_frame": False,
        #     }
        #     rotated_frames.append(row)
        if next_rotation_starts_in_next_frame:
            rotation_index += 1
    else:
        break

rotated_frames = pd.DataFrame(_rotated_frames)
rotated_frames = rotated_frames.reset_index(drop=True)

new_tif_roated_frames: dict[str, list[np.ndarray]] = {
    "50": [],
    "100": [],
    "150": [],
    "200": [],
}
rotation_start_new_idx: dict[str, list[int]] = {
    "50": [],
    "100": [],
    "150": [],
    "200": [],
}
directions: dict[str, list[int]] = {
    "50": [],
    "100": [],
    "150": [],
    "200": [],
}

for (
    f_idx,
    is_rotating,
    rotation_idx,
    rotation_speed,
    direction,
    rotation_start,
) in zip(
    rotated_frames["frame_idx"],
    rotated_frames["is_rotating"],
    rotated_frames["rotation_idx"],
    rotated_frames["rotation_speed"],
    rotated_frames["direction"],
    rotated_frames["starts_in_frame"],
):
    if rotation_idx == "no_rotation":
        new_tif_roated_frames["50"].append(derotated[f_idx])
        new_tif_roated_frames["100"].append(derotated[f_idx])
        new_tif_roated_frames["150"].append(derotated[f_idx])
        new_tif_roated_frames["200"].append(derotated[f_idx])
    elif rotation_start and rotation_speed == 50:
        new_tif_roated_frames["50"].append(derotated[f_idx])
        rotation_start_new_idx["50"].append(len(new_tif_roated_frames["50"]))
        directions["50"].append(direction)
    elif rotation_start and rotation_speed == 100:
        new_tif_roated_frames["100"].append(derotated[f_idx])
        rotation_start_new_idx["100"].append(len(new_tif_roated_frames["100"]))
        directions["100"].append(direction)
    elif rotation_start and rotation_speed == 150:
        new_tif_roated_frames["150"].append(derotated[f_idx])
        rotation_start_new_idx["150"].append(len(new_tif_roated_frames["150"]))
        directions["150"].append(direction)
    elif rotation_start and rotation_speed == 200:
        new_tif_roated_frames["200"].append(derotated[f_idx])
        rotation_start_new_idx["200"].append(len(new_tif_roated_frames["200"]))
        directions["200"].append(direction)
    elif rotation_speed == 50:
        new_tif_roated_frames["50"].append(derotated[f_idx])
    elif rotation_speed == 100:
        new_tif_roated_frames["100"].append(derotated[f_idx])
    elif rotation_speed == 150:
        new_tif_roated_frames["150"].append(derotated[f_idx])
    elif rotation_speed == 200:
        new_tif_roated_frames["200"].append(derotated[f_idx])


for speed in ["50", "100", "150", "200"]:
    path_tif = Path(
        f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/rotation{speed}/masked_rotation_{speed}.tif"
    )
    imsave(path_tif, np.asarray(new_tif_roated_frames[speed]))

    # save rotation_start_new_idx and directions_50 in a csv
    df = pd.DataFrame(columns=["rotation_start_new_idx", "directions"])
    df["rotation_start_new_idx"] = rotation_start_new_idx[speed]
    df["directions"] = directions[speed]
    path_csv = Path(
        f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/rotation{speed}/rotation_start_new_idx.csv"
    )
    df.to_csv(path_csv, index=False)
