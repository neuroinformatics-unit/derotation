from pathlib import Path

import numpy as np
import yaml
from scipy.io import loadmat
from scipy.signal import find_peaks

from derotation.analysis.analog_preprocessing import (
    apply_rotation_direction,
    check_number_of_rotations,
    when_is_rotation_on,
)
from derotation.load_data.read_binary import read_rc2_bin

path_config = "derotation/config.yml"

with open(path_config, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

path = Path(config["path"])
path_randperm = path / "230327_pollen" / "stimlus_randperm.mat"
path = path / "230803_rot_6deg"
path_aux = path / "230803_rot_6deg_1_001.bin"

this_increment = 6

data, dt, chan_names, config = read_rc2_bin(path_aux, config)
data_dict = {chan: data[:, i] for i, chan in enumerate(chan_names)}

pseudo_random = loadmat(path_randperm)
full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
rotation_ticks = data_dict["PI_rotticks"]
full_rotation = data_dict["PI_rotON"]
direction = np.where(full_rotation_blocks_direction, -1, 1)

rotation_on = when_is_rotation_on(full_rotation)
rotation_on, rotation_blocks_idx = apply_rotation_direction(
    rotation_on, direction
)

rotation_ticks_peaks = find_peaks(
    rotation_ticks,
    height=4,
    distance=20,
)[0]
increments_per_rotation = check_number_of_rotations(
    rotation_ticks_peaks, rotation_blocks_idx, 360, this_increment
)

print(increments_per_rotation)
