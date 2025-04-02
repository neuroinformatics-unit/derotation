# %%
from pathlib import Path

import numpy as np
import tiffile

from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.config.load_config import load_config, update_config_paths
from derotation.load_data.custom_data_loaders import get_analog_signals

path_tif = Path(
    "/Users/laura/local_data/230802_CAA_1120182/imaging/rotation_00001.tif"
)
path_aux_stim = "/Users/laura/local_data/230802_CAA_1120182/aux_stim/230802_CAA_1120182_rotation_1_001.bin"

# Load the tif file
data = tiffile.imread(path_tif)

# Load the analog signals
channel_names = [
    "camera",
    "scanimage_frameclock",
    "scanimage_lineclock",
    "photodiode2",
    "PI_rotON",
    "PI_rotticks",
]
frame_clock, line_clock, full_rotation, rotation_ticks = get_analog_signals(
    path_aux_stim, channel_names
)

# frames of interest
start = 0
end = 400

small_data = data[start:end]
# save it here to test the pipeline
tiffile.imwrite("small_data.tif", small_data)

# now let's take the first 5% of the analog signals and save it
percent = int(len(frame_clock) * 0.05)
frame_clock_small = frame_clock[:percent]
line_clock_small = line_clock[:percent]
full_rotation_small = full_rotation[:percent]
rotation_ticks_small = rotation_ticks[:percent]

#  save them in an unique file
np.save(
    "analogs_small.npy",
    [
        frame_clock_small,
        line_clock_small,
        full_rotation_small,
        rotation_ticks_small,
    ],
)

pipeline_type = "full"
config = load_config(pipeline_type)
config = update_config_paths(
    config=config,
    tif_path=Path("small_data.tif"),
    aux_path=Path("analogs_small.npy"),
    dataset_path=Path(""),
    kind=pipeline_type,
    output_folder=".",
)
config["paths_read"]["path_to_randperm"] = Path(
    "/Users/laura/local_data/stimlus_randperm.mat"
)
# Load data and run the pipeline
pipeline = FullPipeline(config)
# pipeline.number_of_rotations = 3
# pipeline.direction = pipeline.direction[:3]
# pipeline.speed = pipeline.speed[:3]
pipeline()
