# %%
from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract
from scipy.io import loadmat

saving_path = Path(
    "/Users/lauraporta/Source/github/neuroinformatics-unit/derotation/examples/figures/pdf/"
)


def neuropil_subtraction(f, f_neu):
    #  use default parameters for all methods
    neuropil_subtraction = NeuropilSubtract()
    neuropil_subtraction.set_F(f, f_neu)
    neuropil_subtraction.fit()

    r = neuropil_subtraction.r

    f_corr = f - r * f_neu

    # kernel values to be changed for 3-photon data
    # median_kernel_long = 1213, median_kernel_short = 23

    dff = 100 * dff_module.compute_dff_windowed_median(f_corr)

    return dff, r


F_path = Path(
    "/Users/lauraporta/local_data/rotation/230822_CAA_1120509/suite2p/plane0/F.npy"
)
f = np.load(F_path)
print(f.shape)

Fneu_path = Path(
    "/Users/lauraporta/local_data/rotation/230822_CAA_1120509/suite2p/plane0/Fneu.npy"
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


path_randperm = Path(
    "/Users/lauraporta/local_data/rotation/stimlus_randperm.mat"
)
pseudo_random = loadmat(path_randperm)
rotation_speed = pseudo_random["stimulus_random"][:, 0]

full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
direction = np.where(
    full_rotation_blocks_direction, -1, 1
)  # 1 is counterclockwise, -1 is clockwise


rotated_frames_path = Path(
    "/Users/lauraporta/local_data/rotation/230822_CAA_1120509/derotated_image_stack_full.csv"
)
rotated_frames = pd.read_csv(rotated_frames_path)
print(rotated_frames.head())
print(rotated_frames.shape)

full_dataframe = pd.concat([dff, rotated_frames], axis=1)


# find where do rotations start
rotation_on = np.diff(full_dataframe["rotation_count"])


def find_zero_chunks(arr):
    zero_chunks = []
    start = None

    for i in range(len(arr)):
        if arr[i] == 0 and start is None:
            start = i
        elif arr[i] != 0 and start is not None:
            zero_chunks.append((start, i - 1))
            start = None

    # Check if the array ends with a chunk of zeros
    if start is not None:
        zero_chunks.append((start, len(arr) - 1))

    return zero_chunks


starts_ends = find_zero_chunks(rotation_on)

frames_before_rotation = 10
# frames_after_rotation = 10

total_len = 70

full_dataframe["rotation_frames"] = np.zeros(len(full_dataframe))
for i, (start, end) in enumerate(starts_ends):
    frame_array = np.arange(total_len)
    column_index_of_rotation_frames = full_dataframe.columns.get_loc(
        "rotation_frames"
    )
    full_dataframe.iloc[
        start
        - frames_before_rotation : total_len
        + start
        - frames_before_rotation,
        column_index_of_rotation_frames,
    ] = frame_array

    #  extend this value of speed and direction to all this range
    this_speed = full_dataframe.loc[start, "speed"]
    this_direction = full_dataframe.loc[start, "direction"]

    full_dataframe.iloc[
        start
        - frames_before_rotation : total_len
        + start
        - frames_before_rotation,
        full_dataframe.columns.get_loc("speed"),
    ] = this_speed
    full_dataframe.iloc[
        start
        - frames_before_rotation : total_len
        + start
        - frames_before_rotation,
        full_dataframe.columns.get_loc("direction"),
    ] = this_direction


#  directtion, change -1 to CCW and 1 to CW
full_dataframe["direction"] = np.where(
    full_dataframe["direction"] == -1, "CCW", "CW"
)


rois_selection = range(len(dff.columns))


# %%
selected_range = (400, 2000)

for roi in rois_selection:
    roi_selected = full_dataframe.loc[
        :, [roi, "rotation_count", "speed", "direction"]
    ]

    fig, ax = plt.subplots(figsize=(27, 5))
    ax.plot(roi_selected.loc[selected_range[0] : selected_range[1], roi])
    ax.set(xlabel="Frames", ylabel="ΔF/F")

    rotation_on = (
        np.diff(
            roi_selected.loc[
                selected_range[0] : selected_range[1], "rotation_count"
            ]
        )
        == 0
    )

    # add label at the beginning of every block of rotations
    #  if the previous was true, do not write the label
    for i, rotation in enumerate(rotation_on):
        if rotation and not rotation_on[i - 1]:
            ax.text(
                i + selected_range[0] + 3,
                -1100,
                f"{int(roi_selected.loc[i + 5 + selected_range[0], 'speed'])}º/s\n{roi_selected.loc[i + 5 + selected_range[0], 'direction']}",
                fontsize=10,
            )

    #  add gray squares when the rotation is happening using the starst_ends
    for start, end in starts_ends:
        if start > selected_range[0] and end < selected_range[1]:
            ax.axvspan(start, end, color="gray", alpha=0.2)

    fps = 6.74
    # change xticks to seconds
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks / fps).astype(int))
    #  change x label
    ax.set(xlabel="Seconds", ylabel="ΔF/F")

    ax.set_xlim(selected_range)
    ax.set_ylim(-1000, 1000)

    # leave some gap between the axis and the plot
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(saving_path / f"dff_example_{roi}.pdf")
    plt.close()


# %%


custom_palette = sns.color_palette("dark:#5A9_r", 4)

for roi in rois_selection:
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    for i, direction in enumerate(["CW", "CCW"]):
        sns.lineplot(
            x="rotation_frames",
            y=roi,
            data=full_dataframe[(full_dataframe["direction"] == direction)],
            hue="speed",
            palette=custom_palette,
            ax=ax[i],
        )
        ax[i].set_title(f"Direction: {direction}")
        ax[i].legend(title="Speed")

        #  remove top and right spines
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)

        # add vertical lines to show the start of the rotation
        #  start is always at 11, end at total len - 10
        ax[i].axvline(x=frames_before_rotation, color="gray", linestyle="--")

        #  change x axis to seconds
        fps = 6.74
        xticks = ax[i].get_xticks()
        ax[i].set_xticks(xticks)
        ax[i].set_xticklabels(np.round(xticks / fps, 1))
        #  change x label
        ax[i].set(xlabel="Seconds", ylabel="ΔF/F")

    plt.savefig(saving_path / f"roi_{roi}_direction_speed.pdf")
    plt.close()
