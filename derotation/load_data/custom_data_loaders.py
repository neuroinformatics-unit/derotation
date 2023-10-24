import numpy as np
from scipy.io import loadmat


def read_randomized_stim_table(path_to_randperm):
    pseudo_random = loadmat(path_to_randperm)
    full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
    direction = np.where(
        full_rotation_blocks_direction, -1, 1
    )  # 1 is counterclockwise, -1 is clockwise

    speed = pseudo_random["stimulus_random"][:, 0]

    return direction, speed


def read_rc2_bin(path_aux, chan_names):
    n_channels = len(chan_names)

    # Read binary data saved by rc2 (int16)
    # Channels along the columns, samples along the rows.
    data = np.fromfile(path_aux, dtype=np.int16)
    data = data.reshape((-1, n_channels))

    # Transform the data.
    # Convert 16-bit integer to volts between -10 and 10.
    data = -10 + 20 * (data + 2**15) / 2**16

    return data, chan_names


def get_analog_signals(path_to_aux, channel_names):
    data, chan_names = read_rc2_bin(path_to_aux, channel_names)

    data_dict = {chan: data[:, i] for i, chan in enumerate(chan_names)}

    frame_clock = data_dict["scanimage_frameclock"]
    line_clock = data_dict["scanimage_lineclock"]
    full_rotation = data_dict["PI_rotON"]
    rotation_ticks = data_dict["PI_rotticks"]

    return frame_clock, line_clock, full_rotation, rotation_ticks
