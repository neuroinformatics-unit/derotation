from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import matplotlib.pyplot as plt
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
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/no_rotation/suite2p/plane0/F.npy"
)
f = np.load(F_path)
print(f.shape)

Fneu_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/no_rotation/suite2p/plane0/Fneu.npy"
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

path_csv = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/no_rotation/end_of_rotation.csv"
)
end_of_rotation_idx = pd.read_csv(path_csv)

dff_roi = pd.DataFrame(
    columns=[
        "signal",
        "after_rotation",
        "rotation_speed",
        "direction",
        "local_frame_idx",
    ]
)
# focus on ROI id = 6
dff_roi["signal"] = dff[6]

# add new column "after_rotation" and "rotation_speed"
dff_roi["after_rotation"] = False

if not end_of_rotation_idx.empty:
    for idx, row in end_of_rotation_idx.iterrows():
        dff_roi.loc[row["frame_idx"] + 1, "after_rotation"] = True
        dff_roi.loc[row["frame_idx"] + 1, "rotation_speed"] = rotation_speed[
            idx
        ]
        dff_roi.loc[row["frame_idx"] + 1, "direction"] = direction[idx]

counter = 0
for idx, row in dff_roi.iterrows():
    if idx == 0:
        dff_roi.loc[idx, "local_frame_idx"] = counter
        continue
    if row["after_rotation"] is True:
        counter = 0
        dff_roi.loc[idx, "local_frame_idx"] = counter
    if row["after_rotation"] is False:
        dff_roi.loc[idx, "rotation_speed"] = dff_roi.loc[
            idx - 1, "rotation_speed"
        ]
        dff_roi.loc[idx, "direction"] = dff_roi.loc[idx - 1, "direction"]
        dff_roi.loc[idx, "local_frame_idx"] = counter
        counter += 1


print(dff_roi.head())
print(dff_roi[dff_roi["after_rotation"] is True].head())


sns.relplot(
    data=dff_roi,
    x="local_frame_idx",
    y="signal",
    hue="rotation_speed",
    col="direction",
    kind="line",
    facet_kws=dict(sharex=False),
)

plt.show()

# save plot as png
path_png = Path("/Users/lauraporta/local_data/rotation/dff_roi_2.png")
plt.savefig(path_png)
