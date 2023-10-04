from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tifffile import imsave

from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline()

pipeline.process_analog_signals()

# rotation speed
path_randperm = Path(
    "/Users/lauraporta/local_data/rotation/stimlus_randperm.mat"
)
pseudo_random = loadmat(path_randperm)
rotation_speed = pseudo_random["stimulus_random"][:, 0]
print(rotation_speed)

# non-rotated image is in pipeline.image_stack
# get index of the frame at which rotation starts
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


print(rotated_frames["frame_idx"].unique())

frames_no_rotation = np.setdiff1d(
    np.arange(pipeline.image_stack.shape[0]),
    rotated_frames["frame_idx"].unique(),
)

new_tiff_no_rotation = pipeline.image_stack[frames_no_rotation, :, :]
# save new tiff
path_tif = Path(
    "/Users/lauraporta/local_data/rotation/230818_pollen_rotation/masked_no_rotation.tif"
)
imsave(path_tif, new_tiff_no_rotation)


# save in a csw the frame id just at the end of the rotation
# as in the new tiff where rotation is removed
end_of_rotation_idx = np.where(np.diff(frames_no_rotation) > 1)[
    0
]  # where there is a gap in the frames
print(end_of_rotation_idx)

# save csv
path_csv = Path(
    "/Users/lauraporta/local_data/rotation/230818_pollen_rotation/end_of_rotation.csv"
)
pd.DataFrame(end_of_rotation_idx).to_csv(path_csv)
