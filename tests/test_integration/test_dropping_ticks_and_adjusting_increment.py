import numpy as np

from derotation.analysis.derotation_pipeline import DerotationPipeline


def test_dropping_ticks_and_adjusting_increment(
    derotation_pipeline: DerotationPipeline,
):
    len(derotation_pipeline.rotation_ticks_peaks)
    derotation_pipeline.drop_ticks_outside_of_rotation()
    len(derotation_pipeline.rotation_ticks_peaks)
    (
        new_increments,
        ticks_per_rotation,
    ) = derotation_pipeline.adjust_rotation_increment()

    assert np.sum(ticks_per_rotation) == len(
        derotation_pipeline.rotation_ticks_peaks
    )
