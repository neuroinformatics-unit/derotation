import numpy as np

from derotation.analysis.full_rotation_pipeline import FullPipeline


def test_get_interpolated_angles(derotation_pipeline: FullPipeline):
    derotation_pipeline.drop_ticks_outside_of_rotation()
    (
        derotation_pipeline.corrected_increments,
        derotation_pipeline.ticks_per_rotation,
    ) = derotation_pipeline.adjust_rotation_increment()

    angles = derotation_pipeline.get_interpolated_angles()
    angles = np.round(angles, 0)

    test_angles = [
        0,
        10,
        21,
        21,
        31,
        31,
        34,
        38,
        41,
        41,
        51,
    ]

    assert np.all(angles[99:110] == test_angles)
