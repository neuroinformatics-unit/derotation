import numpy as np
import pandas as pd
from scipy.io import loadmat

path_to_randperm = "/Users/laura/data/derotation/raw/stimlus_randperm.mat"
pseudo_random = loadmat(path_to_randperm)
full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
direction = np.where(
    full_rotation_blocks_direction, -1, 1
)  # 1 is counterclockwise, -1 is clockwise

speed = pseudo_random["stimulus_random"][:, 0]


path_csv = "/Users/laura/data/derotation/raw/230802_CAA_1120182/\
    derotated_image_stack_CE.csv"
df = pd.read_csv(path_csv)

df["direction"] = np.nan * np.ones(len(df))
df["speed"] = np.nan * np.ones(len(df))
df["rotation_count"] = np.nan * np.ones(len(df))

#  drop rotation on column
df = df.drop(columns=["rotation_on"])

rotation_counter = 0
adding_roatation = False
for i in range(len(df)):
    row = df.loc[i]
    if np.abs(row["rotation_angle"]) > 0.0:
        adding_roatation = True
        row["direction"] = direction[rotation_counter]
        row["speed"] = speed[rotation_counter]
        row["rotation_count"] = rotation_counter

        #  save to df
        df.loc[i] = row

    if i == 149:
        print("debug")
    # if next rotation is 0, increase counter
    # if i < 79 and df.loc[i + 1, 'rotation_angle'] == 0.0:
    if (
        rotation_counter < 79
        and adding_roatation
        and np.abs(df.loc[i + 1, "rotation_angle"]) == 0.0
    ):
        rotation_counter += 1
        adding_roatation = False

#  save csv
df.to_csv(
    "/Users/laura/data/derotation/raw/230802_CAA_1120182/derotated_image_stack_CE_correct.csv"
)
