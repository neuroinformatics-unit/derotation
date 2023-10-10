from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import numpy as np
import pandas as pd
import tifffile as tiff
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract
from matplotlib import pyplot as plt
from scipy.io import loadmat

from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline()

pipeline.process_analog_signals()

# import already rotated images
path_tif = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/derotated/no_background/masked_no_background.tif"
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
        end_rotation = rot_end[rotation_index]
        try:
            next_rotation_starts = rot_start[rotation_index + 1]
        except IndexError:
            next_rotation_starts = rot_end[rotation_index]
        rotation_starts_in_frame = (start_rotation >= f_start) and (
            start_rotation < f_end
        )
        intra_rotaion_frame = (start_rotation < f_start) and (
            end_rotation > f_start
        )
        rotation_ends_in_frame = (end_rotation >= f_start) and (
            end_rotation <= f_end
        )
        inter_rotation_frame = (f_start >= end_rotation) and (
            f_end <= next_rotation_starts
        )
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
        elif intra_rotaion_frame or rotation_ends_in_frame:
            row = {
                "frame_idx": f_idx,
                "is_rotating": True,
                "rotation_idx": rotation_index,
                "rotation_speed": rotation_speed[rotation_index],
                "direction": direction[rotation_index],
                "starts_in_frame": False,
            }
            _rotated_frames.append(row)
        elif inter_rotation_frame:
            row = {
                "frame_idx": f_idx,
                "is_rotating": False,
                "rotation_idx": rotation_index,
                "rotation_speed": rotation_speed[rotation_index],
                "direction": direction[rotation_index],
                "starts_in_frame": False,
            }
            _rotated_frames.append(row)
        if next_rotation_starts_in_next_frame:
            rotation_index += 1
    else:
        break


rotated_frames = pd.DataFrame(_rotated_frames)
rotated_frames = rotated_frames.reset_index(drop=True)

print(rotated_frames.head())
print(rotated_frames.shape)


def neuropil_subtraction(f, f_neu):
    #  use default parameters for all methods
    neuropil_subtraction = NeuropilSubtract()
    neuropil_subtraction.set_F(f, f_neu)
    neuropil_subtraction.fit()

    r = neuropil_subtraction.r

    f_corr = f - r * f_neu

    # kernel values to be changed for 3-photon data
    # median_kernel_long = 1213, median_kernel_short = 23

    dff = 100 * dff_module.compute_dff_windowed_median(
        f_corr, median_kernel_long=1213, median_kernel_short=23
    )

    return dff, r


F_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/derotated/no_background/suite2p/plane0/F.npy"
)
f = np.load(F_path)
print(f.shape)

Fneu_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/derotated/no_background/suite2p/plane0/Fneu.npy"
)
fneu = np.load(Fneu_path)
print(fneu.shape)

dff, r = neuropil_subtraction(
    f=f,
    f_neu=fneu,
)

dff = pd.DataFrame(dff.T)
print(dff.shape)
print(dff.head())

# no neuropil subtraction
# dff = pd.DataFrame(f.T)

#  merge dff with rotated_frames
n_roi = len(dff.columns)

plotting_window = (-30, 70)
for roi in range(n_roi):
    dff_roi = dff.loc[:, roi]

    for speed in [50, 100, 150, 200]:
        #  use rotation speed to select the right frames
        length_this_rotation = (
            len(
                rotated_frames[
                    (rotated_frames.is_rotating)
                    & (rotated_frames.rotation_speed == speed)
                ]
            )
            // np.where(rotation_speed == speed)[0].shape[0]
        )
        beginning_of_rotation = -plotting_window[0]
        end_of_rotation = length_this_rotation + beginning_of_rotation

        rotation_data = {}
        for direction in [-1, 1]:
            starting_frames = rotated_frames.loc[
                (rotated_frames["rotation_speed"] == speed)
                & (rotated_frames["starts_in_frame"] is True)
                & (rotated_frames["direction"] == direction)
            ]

            start = starting_frames["frame_idx"].values + plotting_window[0]
            end = starting_frames["frame_idx"].values + plotting_window[1]
            idx_windows = [np.arange(s, e) for s, e in zip(start, end)]
            idx_windows = np.concatenate(idx_windows)

            # make a numpy array of the same size as the window
            roi_window_dff = np.zeros(len(idx_windows))
            # fill it with the dff values
            roi_window_dff[:] = dff_roi.iloc[idx_windows]
            # reshape
            roi_window_dff = roi_window_dff.reshape(
                len(start), len(idx_windows) // len(start)
            ).T
            #  make it as a dataframe and add info
            # on if the rotation is on or not
            # roi_window_dff = pd.DataFrame(roi_window_dff.T)
            rotation_data[direction] = roi_window_dff

        fig, ax = plt.subplots(figsize=(10, 5))
        # make a line plot with matplotlib with all the dff values

        ax.plot(rotation_data[1], color="red", alpha=0.1)
        median_cw = np.median(rotation_data[1], axis=1)
        ax.plot(median_cw, color="red", alpha=1)

        ax.plot(rotation_data[-1], color="green", alpha=0.1)
        median_ccw = np.median(rotation_data[-1], axis=1)
        ax.plot(median_ccw, color="green", alpha=1)

        ax.axvline(x=beginning_of_rotation, color="black")
        ax.axvline(x=end_of_rotation, color="black")
        # a.set_ylim(-3700, -2900)

        # add color legend
        ax.annotate(
            "counterclockwise",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            color="green",
        )
        ax.annotate(
            "clockwise", xy=(0.05, 0.9), xycoords="axes fraction", color="red"
        )

        fig.suptitle(f"roi {roi} speed {speed}")
        #  save this plot
        fig.savefig(
            f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/derotated/no_background/plots/neuropil_subtraction/roi_{roi}_speed_{speed}.png"
        )
