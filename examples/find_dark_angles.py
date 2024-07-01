# %%
from pathlib import Path

import allensdk.brain_observatory.dff as dff_module
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from allensdk.brain_observatory.r_neuropil import NeuropilSubtract

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

    dff = 100 * dff_module.compute_dff_windowed_median(
        f_corr, median_kernel_long=1213, median_kernel_short=23
    )

    return dff, r


F_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/incremental/derotated/suite2p/plane0/F.npy"
)
f = np.load(F_path)
print(f.shape)

Fneu_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/incremental/derotated/suite2p/plane0/Fneu.npy"
)
fneu = np.load(Fneu_path)
print(fneu.shape)

dff, r = neuropil_subtraction(
    f=f,
    f_neu=fneu,
)

dff = pd.DataFrame(dff.T)
# dff = pd.DataFrame(f.T)
print(dff.shape)
print(dff.head())


rotation_times_path = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/incremental/derotated/derotated_incremental_image_stack_CE.csv"
)
rotation_times = pd.read_csv(rotation_times_path)


# dff = pd.melt(dff, value_vars=dff.columns, var_name='roi', value_name='df/f')
n_roi = len(dff.columns)

# dff = (dff - dff.mean()) / dff.std()

dff = dff.rolling(20).sum()


dff["rotation_angle"] = rotation_times["rotation_angle"]

# tollerace_deg = 1
# for angle in range(0, 360, 10):
#     dff = dff[(dff["rotation_angle"] - angle) < tollerace_deg]

# dff_rotating_times = dff[(dff['rotation_angle'] > 0) | (dff['rotation_angle'] < 0)]
dff["rotation_angle"] = np.deg2rad(dff["rotation_angle"])


# %%
fig, ax = plt.subplots(3, 4, subplot_kw={"projection": "polar"})
for i in range(12):
    h = i // 3
    w = i % 3
    ax[w, h].plot(dff["rotation_angle"], dff[i])

    # subtitle is i
    ax[w, h].set_title(i)
# plt.show()

plt.savefig(saving_path / "fig1.png")

# dff = (dff - dff.mean()) / dff.std()

# %%
fig2, ax2 = plt.subplots(3, 4, subplot_kw={"projection": "polar"})
tollerace_deg = 1
for i in range(12):
    h = i // 3
    w = i % 3

    mean_vals = []
    dff["rotation_angle"] = np.rad2deg(dff["rotation_angle"])

    for angle in range(0, 360, 10):
        mean_vals.append(
            np.mean(dff[(dff["rotation_angle"] - angle) < tollerace_deg][i])
        )

    dff["rotation_angle"] = np.deg2rad(dff["rotation_angle"])

    angles = np.deg2rad(list(range(0, 360, 10)))
    ax2[w, h].scatter(angles, mean_vals, s=1)

    # subtitle is i
    ax2[w, h].set_title(i)


# plt.show()
plt.savefig(saving_path / "fig2.png")

# %%

tollerace_deg = 1
for i in range(12):
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
    mean_vals = []
    dff["rotation_angle"] = np.rad2deg(dff["rotation_angle"])

    for angle in range(0, 360, 10):
        mean_vals.append(
            np.mean(dff[(dff["rotation_angle"] - angle) < tollerace_deg][i])
        )

    dff["rotation_angle"] = np.deg2rad(dff["rotation_angle"])

    angles = np.deg2rad(list(range(0, 360, 10)))
    ax.plot(angles, mean_vals)
    #  fill in the area below the curve

    # ax.fill_between(angles, mean_vals, 0, alpha=0.5)
    ax.set_title(i)
    #  set min and max r
    minimum = np.min(mean_vals)
    maximum = np.max(mean_vals)
    ax.set_ylim(-200, 200)

    # remove grid
    ax.grid(False)
    #  remove r labels
    ax.set_yticklabels([])
    plt.savefig(saving_path / f"angle_roi_{i}.png")

# %%
