from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import numpy as np
import pandas as pd
import seaborn as sns
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract
from scipy.io import loadmat


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

subset = full_dataframe[
    (full_dataframe["speed"] == 100) & (full_dataframe["direction"] == -1)
]

rois_selection = [4, 8, 14, 20, 23]

merged_mean = pd.DataFrame()
for roi in rois_selection:
    mean_response = subset.loc[:, [roi, "rotation_count"]]
    mean_response["counter"] = np.zeros(len(mean_response)) - 1
    latest_rotation = -1
    for idx in mean_response.index:
        if mean_response.loc[idx, "rotation_count"] > latest_rotation:
            counter = 0
            latest_rotation = mean_response.loc[idx, "rotation_count"]
        elif mean_response.loc[idx, "rotation_count"] == latest_rotation:
            counter += 1
        mean_response.loc[idx, "counter"] = counter

    mean = mean_response.groupby("counter").mean()

    # plt.plot(mean.loc[:, [roi]])

    merged_mean[f"roi_{roi}"] = mean.loc[:, [roi]]

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
ax = sns.lineplot(merged_mean)
ax.set(xlabel="Frames during rotation (100 deg/s)", ylabel="ΔF/F")


print("debug")
