from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)

derotator_incremental = IncrementalPipeline("incremental_rotation")
derotator_incremental()
derotate = FullPipeline("full_rotation")
derotate.center_of_rotation = derotator_incremental.center_of_rotation
derotate()
