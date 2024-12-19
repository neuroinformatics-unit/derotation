from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)
from derotation.plotting_hooks.for_derotation import (
    image_completed,
    line_addition,
)
import numpy as np

derotator_incremental = IncrementalPipeline("incremental_rotation")
derotator_incremental()
hooks = {
    # "plotting_hook_line_addition": line_addition,
    "plotting_hook_image_completed": image_completed,
}
derotate = FullPipeline("full_rotation")
derotate.hooks = hooks
derotate.center_of_rotation = derotator_incremental.center_of_rotation

ellipse_fits = derotator_incremental.all_ellipse_fits
if ellipse_fits["a"] < ellipse_fits["b"]:
    print("a < b")
    rotation_plane_angle = np.degrees(
        np.arccos(ellipse_fits["a"] / ellipse_fits["b"])
    )
    rotation_plane_orientation = np.degrees(ellipse_fits["theta"])
else:
    print("a > b")
    rotation_plane_angle = np.degrees(
        np.arccos(ellipse_fits["b"] / ellipse_fits["a"])
    )
    theta = ellipse_fits["theta"] + np.pi / 2
    rotation_plane_orientation = np.degrees(theta)

print(
    f"rotation_plane_angle: {rotation_plane_angle}, rotation_plane_orientation: {rotation_plane_orientation}"
)
rotation_plane_angle = np.round(rotation_plane_angle, 1)
rotation_plane_orientation = np.round(
    rotation_plane_orientation, 1
)
derotate.rotation_plane_angle = rotation_plane_angle
derotate.rotation_plane_orientation = rotation_plane_orientation
derotate()
