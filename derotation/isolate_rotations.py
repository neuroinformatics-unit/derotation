from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.io import loadmat
from tifffile import imsave

from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline()

pipeline.process_analog_signals()

# import already rotated images
path_tif = Path("/Users/laura/data/derotation/pre-processed/CAA_2/masked.tif")
derotated = tiff.imread(path_tif)

# rotation speed
path_randperm = Path("/Users/laura/data/derotation/raw/stimlus_randperm.mat")
pseudo_random = loadmat(path_randperm)
rotation_speed = pseudo_random["stimulus_random"][:, 0]
print(rotation_speed)

# in rotation ticks time
frame_start, frame_end, _ = pipeline.get_starting_and_ending_times(
    clock_type="frame"
)

rotated_frames = pd.DataFrame(
    columns=["frame_idx", "is_rotating", "rotation_idx", "rotation_speed"]
)
rot_start = pipeline.rot_blocks_idx["start"]
rot_end = pipeline.rot_blocks_idx["end"]
for rot_idx, (start_rotation, end_rotation) in enumerate(
    zip(rot_start, rot_end)
):
    for f_idx, (f_start, f_end) in enumerate(zip(frame_start, frame_end)):
        rotation_starts_in_frame = (
            start_rotation >= f_start and start_rotation <= f_end
        )
        rotation_already_started = (
            start_rotation < f_start and end_rotation > f_start
        )
        rotation_ends_in_frame = (
            end_rotation >= f_start and end_rotation <= f_end
        )
        if (
            rotation_starts_in_frame
            or rotation_already_started
            or rotation_ends_in_frame
        ):
            row = {
                "frame_idx": f_idx,
                "is_rotating": True,
                "rotation_idx": rot_idx,
                "rotation_speed": rotation_speed[rot_idx],
            }
            df = pd.DataFrame(row, index=[0])
            rotated_frames = pd.concat([rotated_frames, df])

for speed in np.unique(rotation_speed):
    print(speed)
    frames_rotating_at_speed = rotated_frames[
        rotated_frames["rotation_speed"] == speed
    ]["frame_idx"].values
    derotated_selected = derotated[frames_rotating_at_speed.astype(int)]
    imsave(
        f"/Users/laura/data/derotation/pre-processed/CAA_2/derotated_{speed}.tif",
        derotated_selected,
    )
