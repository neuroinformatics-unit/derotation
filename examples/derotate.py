from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)
from derotation.plotting_hooks.for_derotation import (
    image_completed,
)

derotator_incremental = IncrementalPipeline("incremental_rotation")
derotator_incremental()
hooks = {
    # "plotting_hook_line_addition": line_addition,
    "plotting_hook_image_completed": image_completed,
}
derotate = FullPipeline("full_rotation")
derotate.hooks = hooks
derotate.center_of_rotation = derotator_incremental.center_of_rotation
derotate()
