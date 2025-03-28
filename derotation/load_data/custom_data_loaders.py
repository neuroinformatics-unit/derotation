"""
These functions are used to load data from the custom data format used in the
experiments. The data is saved by MATLAB scripts that are not included in
this repository.
The idea is that these functions can be re-written if the preprocessing in
Matlab is changed or if the experimental setup is changed.
"""

import numpy as np
from scipy.io import loadmat


def read_randomized_stim_table(path_to_randperm: str) -> tuple:
    """Read the randomized stimulus table used in the experiments.
    It contains the direction and speed of rotation for each trial.

    Parameters
    ----------
    path_to_randperm : str
        Path to the randomized stimulus table.

    Returns
    -------
    tuple
        Tuple containing the direction and speed of rotation for each trial.
    """
    pseudo_random = loadmat(path_to_randperm)
    full_rotation_blocks_direction = pseudo_random["stimulus_random"][:, 2] > 0
    direction = np.where(
        full_rotation_blocks_direction, -1, 1
    )  # 1 is counterclockwise, -1 is clockwise

    speed = pseudo_random["stimulus_random"][:, 0]

    return direction, speed


def convert_to_volts(data: np.ndarray) -> np.ndarray:
    """Convert 16-bit integer to volts between -10 and 10.

    Parameters
    ----------
    data : np.ndarray
        Data from the binary file.

    Returns
    -------
    np.ndarray
        Data in volts.
    """
    data = -10 + 20 * (data + 2**15) / 2**16
    return data


def get_analog_signals(path_to_aux: str, channel_names: list) -> tuple:
    """Read the analog signals: frame clock, line clock, full rotation and
    rotation ticks.

    Parameters
    ----------
    path_to_aux : str
        Path to the binary file.
    channel_names : list
        Names of the channels in the binary file.

    Returns
    -------
    tuple
        Tuple containing the frame clock, line clock, full rotation and
        rotation ticks.
    """

    data = np.fromfile(path_to_aux, dtype=np.int16)  # Has to be read as int16
    data = data.astype(np.int32)  # cast data to int32 to avoid overflow
    data = data.reshape((-1, len(channel_names)))
    data = convert_to_volts(data)

    data_dict = {chan: data[:, i] for i, chan in enumerate(channel_names)}

    frame_clock = data_dict["scanimage_frameclock"]
    line_clock = data_dict["scanimage_lineclock"]
    full_rotation = data_dict["PI_rotON"]
    rotation_ticks = data_dict["PI_rotticks"]

    return frame_clock, line_clock, full_rotation, rotation_ticks
