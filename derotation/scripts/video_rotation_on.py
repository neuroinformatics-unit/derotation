from pathlib import Path

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from tqdm import tqdm

from derotation.analysis.derotation_pipeline import DerotationPipeline

pipeline = DerotationPipeline()

pipeline.process_analog_signals()

# import already rotated images
path_tif = Path(
    "/Users/lauraporta/local_data/rotation/230802_CAA_1120182/derotated/no_background/masked_no_background.tif"
)
derotated = tiff.imread(path_tif)
min_value = np.nanmin(derotated[0])
max_value = np.nanmax(derotated[0])

lines_per_img = derotated.shape[1]

for i, img in tqdm(enumerate(derotated), total=derotated.shape[0]):
    fig, ax = plt.subplots(figsize=(10, 10))
    line_id = i * lines_per_img
    angle = pipeline.rot_deg_line[line_id]

    #  if angle is almost 0, set it to 0
    if abs(angle) < 0.001:
        angle = 0
    ax.imshow(
        img,
        cmap="turbo",
        vmin=min_value,
        vmax=max_value,
    )
    ax.annotate(
        f"Angle: {angle:.2f}°",
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=20,
        color="white",
    )
    ax.axis("off")
    fig.savefig(
        f"/Users/lauraporta/local_data/rotation/230802_CAA_1120182/derotated/no_background/frames_for_video/derotated_{i}.png",
        bbox_inches="tight",
        dpi=300,
    )
    ax.cla()
    plt.close(fig)
