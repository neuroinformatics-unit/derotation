from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import numpy as np
import pandas as pd
import seaborn as sns
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract


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


for speed in ["50", "100", "150"]:
    print(f"Analyzing speed {speed}")
    F_path = Path(
        f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/rotation{speed}/suite2p/plane0/F.npy"
    )
    f = np.load(F_path)
    print(f.shape)

    Fneu_path = Path(
        f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/rotation{speed}/suite2p/plane0/Fneu.npy"
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

    path_csv = Path(
        f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/rotation{speed}/rotation_start_new_idx.csv"
    )
    rotation_start_new_idx = pd.read_csv(path_csv)

    for roi_id in range(len(dff.columns)):
        dff_roi = pd.DataFrame(
            columns=[
                "signal",
                "after_rotation",
                "direction",
                "local_frame_idx",
            ]
        )

        dff_roi["signal"] = dff[roi_id]

        # add new column "after_rotation" and "rotation_speed"
        dff_roi["after_rotation"] = False

        for idx, row in rotation_start_new_idx.iterrows():
            dff_roi.loc[
                row["rotation_start_new_idx"] + 1, "after_rotation"
            ] = True
            dff_roi.loc[row["rotation_start_new_idx"] + 1, "direction"] = row[
                "directions"
            ]

        shift = -10
        counter = 0
        for idx, row in dff_roi.iterrows():
            if idx == 0:
                dff_roi.loc[idx, "local_frame_idx"] = counter + shift
                continue
            if row["after_rotation"] is True:
                counter = 0
                dff_roi.loc[idx, "local_frame_idx"] = counter + shift
            if row["after_rotation"] is False:
                dff_roi.loc[idx, "direction"] = dff_roi.loc[
                    idx - 1, "direction"
                ]
                dff_roi.loc[idx, "local_frame_idx"] = counter + shift
                counter += 1

        print(dff_roi.head())
        print(dff_roi[dff_roi["after_rotation"] is True].head())

        g = sns.relplot(
            data=dff_roi,
            x="local_frame_idx",
            y="signal",
            col="direction",
            kind="line",
            facet_kws=dict(sharex=False),
        )
        #  add vertical gray box when after rotation is False
        for ax in g.axes.flat:
            for idx, row in dff_roi.iterrows():
                if row["after_rotation"] is False:
                    ax.axvspan(
                        row["local_frame_idx"],
                        row["local_frame_idx"] + 1,
                        facecolor="gray",
                        alpha=0.5,
                    )

        # save plot as png
        path_png = Path(
            f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/rotation{speed}/roi_{roi_id}.png"
        )
        g.savefig(path_png)
